import cv2
import numpy as np
import time
from gpiozero import Motor, Device
from gpiozero.pins.pigpio import PiGPIOFactory  # pip install pigpio; sudo pigpiod

# ====== System/OCV hints ======
cv2.setUseOptimized(True)
cv2.setNumThreads(1)

# ====== GPIO PWM backend (lower CPU than software PWM) ======
try:
    Device.pin_factory = PiGPIOFactory()  # comment out if you don't run pigpiod
except Exception:
    pass  # fall back to default if pigpio isn't running

# ====== CAMERA & TRACKBARS ======
def empty(a): pass

def initTrackbars():
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 520, 260)
    cv2.createTrackbar("Hue Min", "TrackBars", 0,   179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 0,   255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 0,   255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Motor (0=Stop,1=Start)", "TrackBars", 0, 1, empty)

def getTrackbarValues():
    hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
    smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
    vmin = cv2.getTrackbarPos("Val Min", "TrackBars")
    hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
    smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
    vmax = cv2.getTrackbarPos("Val Max", "TrackBars")
    motor_on = cv2.getTrackbarPos("Motor (0=Stop,1=Start)", "TrackBars") == 1
    return [hmin, smin, vmin], [hmax, smax, vmax], motor_on

# Pre-allocate kernel once
KERNEL = np.ones((3, 3), np.uint8)

def findColor(img, lower, upper):
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.erode(mask, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=1)
    colorImg = cv2.bitwise_and(img, img, mask=mask)
    return mask, colorImg

def getLargestContour(mask, min_area=300):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area: return None, None, None
    (x, y), radius = cv2.minEnclosingCircle(c)
    x, y, radius = int(x), int(y), int(radius)
    M = cv2.moments(c); m00 = M["m00"] if M["m00"] else 1
    cx, cy = int(M["m10"]/m00), int(M["m01"]/m00)
    return (x, y, radius), (cx, cy), c

# ====== MOTORS (pins per Oivio Pi) ======
left_motor  = Motor(forward=20, backward=9, pwm=True)
right_motor = Motor(forward=6,  backward=5, pwm=True)

def drive(left_speed, right_speed):
    left_speed  = max(-1.0, min(1.0, left_speed))
    right_speed = max(-1.0, min(1.0, right_speed))
    if left_speed >= 0:  left_motor.forward(left_speed)
    else:                left_motor.backward(-left_speed)
    if right_speed >= 0: right_motor.forward(right_speed)
    else:                right_motor.backward(-right_speed)

def stop():
    left_motor.stop(); right_motor.stop()

# ====== CONTROL GAINS & LIMITS ======
K_drive  = 0.010
FWD_BIAS = 0.12
MAX_SPEED= 0.35
DESIRED_RADIUS = 60
NO_TARGET_BRAKE_SEC = 0.4

X_DEADBAND   = 8
STEER_LIMIT  = 0.55
SMOOTH_ALPHA = 0.60

# --- Smooth, centered tracking settings ---
CENTER_BAND_BASE = 24      # px: accept this much off-center with no steering
CENTER_BAND_GROW = 0.6     # grow band when far (smaller target) for extra chill
TURN_KP = 0.0038           # softer turns
TURN_KD = 0.0020           # damp oscillation
TURN_ALPHA = 0.6           # low-pass on turn command (0..1), higher = smoother
CMD_SLEW = 0.08            # max change per cycle for each wheel command

# Optional: never reverse when "too close" unless it's *very* close
REVERSE_ONLY_IF_CLOSER_THAN = 30  # pixels; set 0 to allow reversing any time

# ====== MAIN LOOP ======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  424)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)  # may or may not be honored
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # if supported

initTrackbars()
last_seen_t = time.time()
last_t = time.time()
ema_x_err = None
prev_motor_on = False

# Rate limiting
CONTROL_HZ = 30.0
CONTROL_PERIOD = 1.0 / CONTROL_HZ
last_control = 0.0
frame_i = 0

# State for smoothing
prev_x_eff = 0.0
turn_lp = 0.0
left_cmd, right_cmd = 0.0, 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_i += 1
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        now = time.time()
        do_control = (now - last_control) >= CONTROL_PERIOD

        # Read sliders
        (hmin,smin,vmin),(hmax,smax,vmax),motor_on = getTrackbarValues()
        lower, upper = [hmin,smin,vmin], [hmax,smax,vmax]

        # HSV, mask, contour
        mask, imgColor = findColor(frame, lower, upper)
        circle, center, contour = getLargestContour(mask, min_area=500)

        H, W = frame.shape[:2]
        cx_target = W // 2

        # Handle motor toggle edges
        if motor_on and not prev_motor_on:
            ema_x_err = None
            prev_x_eff = 0.0
            turn_lp = 0.0
            left_cmd = right_cmd = 0.0
            last_t = now
        elif not motor_on and prev_motor_on:
            stop()
        prev_motor_on = motor_on

        if center and motor_on and do_control:
            last_control = now
            last_seen_t = now

            # Annotations (only when controlling to reduce UI cost)
            if contour is not None:
                cv2.drawContours(display, [contour], -1, (0,255,0), 2)
                cv2.circle(display, (center[0], center[1]), 5, (0,0,255), -1)
            if circle:
                cv2.circle(display, (circle[0], circle[1]), circle[2], (0,255,0), 2)

            # --- Smooth, soft-band steering ---
            x_err_raw = center[0] - cx_target
            det_r     = (circle[2] or 0)
            r_err     = DESIRED_RADIUS - det_r  # + if too far

            # Dynamic soft dead-zone
            dyn_band = int(
                CENTER_BAND_BASE +
                CENTER_BAND_GROW * max(0, DESIRED_RADIUS - det_r) / max(1, DESIRED_RADIUS) * CENTER_BAND_BASE
            )

            # Smooth horizontal error (EMA)
            ema_x_err = x_err_raw if ema_x_err is None else int(
                SMOOTH_ALPHA * ema_x_err + (1 - SMOOTH_ALPHA) * x_err_raw
            )

            # Apply soft dead-zone
            if abs(ema_x_err) <= dyn_band:
                x_eff = 0
            else:
                x_eff = (abs(ema_x_err) - dyn_band) * (1 if ema_x_err > 0 else -1)

            # PD steering with extra low-pass
            dt  = max(1e-3, now - last_t)
            last_t = now
            d_err = (x_eff - prev_x_eff) / dt
            prev_x_eff = x_eff

            turn_raw = TURN_KP * x_eff + TURN_KD * d_err
            turn_lp = TURN_ALPHA * turn_lp + (1 - TURN_ALPHA) * turn_raw
            turn = max(-STEER_LIMIT, min(STEER_LIMIT, turn_lp))

            # --- Forward/back ---
            # Optional safety: don't reverse unless very close
            if REVERSE_ONLY_IF_CLOSER_THAN > 0 and det_r > (DESIRED_RADIUS + REVERSE_ONLY_IF_CLOSER_THAN):
                # way too close -> allow reverse
                fwd  = K_drive * r_err
            else:
                # never reverse: clamp r_err to >= 0
                fwd_r_err = max(0, r_err)
                fwd = K_drive * fwd_r_err

            # Keep a small forward bias when far
            if r_err > 0:
                fwd += FWD_BIAS

            # De-emphasize forward when off-center
            offset_factor = max(0.35, 1.0 - abs(x_eff) / (W // 2))
            fwd *= offset_factor

            # --- Mix to differential drive and slew-limit for smoothness ---
            left_cmd_target  = max(-MAX_SPEED, min(MAX_SPEED,  fwd - turn))
            right_cmd_target = max(-MAX_SPEED, min(MAX_SPEED,  fwd + turn))

            def slew(cur, tgt, step):
                if tgt > cur:  return min(tgt, cur + step)
                else:          return max(tgt, cur - step)

            left_cmd  = slew(left_cmd,  left_cmd_target,  CMD_SLEW)
            right_cmd = slew(right_cmd, right_cmd_target, CMD_SLEW)

            drive(left_cmd, right_cmd)

            cv2.putText(display,
                        f"x:{int(ema_x_err)} r:{int(r_err)} band:{dyn_band} L:{left_cmd:.2f} R:{right_cmd:.2f}",
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

        else:
            # Either motor is off or no target or holding control rate
            if not motor_on:
                stop()
                cv2.putText(display, "Motor: STOP", (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                if time.time() - last_seen_t > NO_TARGET_BRAKE_SEC:
                    stop(); ema_x_err = None; prev_x_eff = 0.0; turn_lp = 0.0
                cv2.putText(display, "No target", (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # GUI – update heavy windows less often
        cv2.imshow("Tracking+Control", display)
        if frame_i % 4 == 0:
            cv2.imshow("Mask", mask)
            cv2.imshow("Detected Colour", imgColor)

        key = cv2.waitKey(2) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            cur = cv2.getTrackbarPos("Motor (0=Stop,1=Start)", "TrackBars")
            cv2.setTrackbarPos("Motor (0=Stop,1=Start)", "TrackBars", 0 if cur else 1)

finally:
    stop()
    cap.release()
    cv2.destroyAllWindows()
