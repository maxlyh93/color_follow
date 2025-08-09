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

class PID:
    def __init__(self, kp=0.0045, ki=0.0, kd=0.0025, i_clamp=0.25):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i, self.prev = 0.0, None
        self.i_clamp = i_clamp
    def reset(self):
        self.i, self.prev = 0.0, None
    def update(self, err, dt):
        if dt <= 0: dt = 1e-3
        d = 0.0 if self.prev is None else (err - self.prev) / dt
        self.i = max(-self.i_clamp, min(self.i_clamp, self.i + err * dt))
        self.prev = err
        return self.kp * err + self.ki * self.i + self.kd * d

pid_turn = PID(kp=0.0045, ki=0.0, kd=0.0025)

# ====== MAIN LOOP ======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  424)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)               # may or may not be honored
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

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_i += 1
        frame = cv2.flip(frame, 1)

        # Skip heavy work to maintain budget
        now = time.time()
        do_control = (now - last_control) >= CONTROL_PERIOD

        # Only read sliders occasionally (cheap anyway, but this reduces GUI chatter)
        (hmin,smin,vmin),(hmax,smax,vmax),motor_on = getTrackbarValues()
        lower, upper = [hmin,smin,vmin], [hmax,smax,vmax]

        # Compute mask every other frame to halve HSV work
        if frame_i % 2 == 0 or do_control:
            mask, imgColor = findColor(frame, lower, upper)
        else:
            # fallback if skipped (rare)
            mask, imgColor = findColor(frame, lower, upper)

        circle, center, contour = getLargestContour(mask, min_area=500)

        H, W = frame.shape[:2]
        cx_target = W // 2

        # Edge transitions for the motor toggle
        if motor_on and not prev_motor_on:
            pid_turn.reset(); ema_x_err = None; last_t = now
        elif not motor_on and prev_motor_on:
            stop()
        prev_motor_on = motor_on

        display = frame.copy()

        if center and motor_on and do_control:
            last_control = now
            last_seen_t = now

            # Annotations (only when we control—less drawing)
            if contour is not None:
                cv2.drawContours(display, [contour], -1, (0,255,0), 2)
                cv2.circle(display, (center[0], center[1]), 5, (0,0,255), -1)
            if circle:
                cv2.circle(display, (circle[0], circle[1]), circle[2], (0,255,0), 2)

            x_err_raw = center[0] - cx_target
            r_err     = DESIRED_RADIUS - (circle[2] or 0)

            # Smooth + deadband
            ema_x_err = x_err_raw if ema_x_err is None else (
                SMOOTH_ALPHA * ema_x_err + (1 - SMOOTH_ALPHA) * x_err_raw
            )
            x_err = 0 if abs(ema_x_err) < X_DEADBAND else ema_x_err

            dt  = now - last_t; last_t = now
            turn = pid_turn.update(x_err, dt)
            turn = max(-STEER_LIMIT, min(STEER_LIMIT, turn))

            fwd  = K_drive * r_err + (FWD_BIAS if r_err > 0 else 0.0)
            offset_factor = max(0.2, 1.0 - abs(x_err) / (W // 2))
            fwd *= offset_factor

            left_cmd  = max(-MAX_SPEED, min(MAX_SPEED,  fwd - turn))
            right_cmd = max(-MAX_SPEED, min(MAX_SPEED,  fwd + turn))
            drive(left_cmd, right_cmd)

            cv2.putText(display, f"RUN x:{int(x_err)} r:{r_err} L:{left_cmd:.2f} R:{right_cmd:.2f}",
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

        else:
            if not motor_on:
                stop()
                cv2.putText(display, "Motor: STOP", (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                if time.time() - last_seen_t > NO_TARGET_BRAKE_SEC:
                    stop(); pid_turn.reset(); ema_x_err = None
                cv2.putText(display, "No target", (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # GUI – update heavy windows less often
        cv2.imshow("Tracking+Control", display)
        if frame_i % 4 == 0:
            cv2.imshow("Mask", mask)
            cv2.imshow("Detected Colour", imgColor)

        key = cv2.waitKey(2) & 0xFF  # tiny delay eases CPU
        if key == ord('q'):
            break
        if key == ord('m'):
            cur = cv2.getTrackbarPos("Motor (0=Stop,1=Start)", "TrackBars")
            cv2.setTrackbarPos("Motor (0=Stop,1=Start)", "TrackBars", 0 if cur else 1)

finally:
    stop()
    cap.release()
    cv2.destroyAllWindows()
