import cv2
import numpy as np
import time
from gpiozero import Motor

# ====== CAMERA & TRACKBARS ======

def empty(a):
    pass

def initTrackbars():
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0,   179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 0,   255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 0,   255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

def getTrackbarValues():
    hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
    smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
    vmin = cv2.getTrackbarPos("Val Min", "TrackBars")
    hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
    smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
    vmax = cv2.getTrackbarPos("Val Max", "TrackBars")
    return [hmin, smin, vmin], [hmax, smax, vmax]

def findColor(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    colorImg = cv2.bitwise_and(img, img, mask=mask)
    return mask, colorImg

def getLargestContour(mask, min_area=400):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, None, None
    (x, y), radius = cv2.minEnclosingCircle(c)
    x, y, radius = int(x), int(y), int(radius)
    M = cv2.moments(c)
    cx = int(M["m10"]/M["m00"]) if M["m00"] else x
    cy = int(M["m01"]/M["m00"]) if M["m00"] else y
    return (x, y, radius), (cx, cy), c

# ====== MOTORS (pins per Oivio Pi) ======
left_motor  = Motor(forward=20, backward=9, pwm=True)
right_motor = Motor(forward=6,  backward=5, pwm=True)

def drive(left_speed, right_speed):
    """Speeds in [-1.0, 1.0]; +forward, -backward"""
    left_speed  = max(-1.0, min(1.0, left_speed))
    right_speed = max(-1.0, min(1.0, right_speed))

    if left_speed >= 0:  left_motor.forward(left_speed)
    else:                left_motor.backward(-left_speed)

    if right_speed >= 0: right_motor.forward(right_speed)
    else:                right_motor.backward(-right_speed)

def stop():
    left_motor.stop()
    right_motor.stop()

# ====== CONTROL GAINS & LIMITS ======
# Distance/approach control (keep object at desired size)
K_drive  = 0.010     # radius error -> forward/back strength
FWD_BIAS = 0.15      # base bias to overcome static friction
MAX_SPEED= 0.40      # cap for stability
DESIRED_RADIUS = 60  # pixels (tune to your camera/height)
NO_TARGET_BRAKE_SEC = 0.5

# --- Centering helpers (PID steering) ---
X_DEADBAND   = 8       # pixels tolerated around center
STEER_LIMIT  = 0.60    # absolute cap for turn command
SMOOTH_ALPHA = 0.60    # EMA filter for x_err (0..1)

class PID:
    def __init__(self, kp=0.0045, ki=0.0, kd=0.0025, i_clamp=0.25):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i, self.prev = 0.0, None
        self.i_clamp = i_clamp
    def reset(self):
        self.i, self.prev = 0.0, None
    def update(self, err, dt):
        d = 0.0 if self.prev is None else (err - self.prev) / max(dt, 1e-3)
        self.i = max(-self.i_clamp, min(self.i_clamp, self.i + err * dt))
        self.prev = err
        return self.kp * err + self.ki * self.i + self.kd * d

pid_turn = PID(kp=0.0045, ki=0.0, kd=0.0025)

# ====== MAIN LOOP ======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

initTrackbars()
last_seen_t = time.time()
last_t = time.time()
ema_x_err = None  # for smoothing horizontal error

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        (hmin, smin, vmin), (hmax, smax, vmax) = getTrackbarValues()
        lower, upper = [hmin, smin, vmin], [hmax, smax, vmax]
        mask, imgColor = findColor(frame, lower, upper)
        circle, center, contour = getLargestContour(mask, min_area=600)

        H, W = frame.shape[:2]
        cx_target = W // 2

        if center:
            last_seen_t = time.time()

            # Draw annotations
            cv2.drawContours(display, [contour], -1, (0,255,0), 2)
            cv2.circle(display, (center[0], center[1]), 5, (0,0,255), -1)
            if circle:
                cv2.circle(display, (circle[0], circle[1]), circle[2], (0,255,0), 2)

            # --- Control errors ---
            x_err_raw = center[0] - cx_target               # + if target is to the right
            r_err     = DESIRED_RADIUS - (circle[2] or 0)   # + if too far away

            # Smooth + deadband horizontal error
            ema_x_err = x_err_raw if ema_x_err is None else int(
                SMOOTH_ALPHA * ema_x_err + (1 - SMOOTH_ALPHA) * x_err_raw
            )
            x_err = 0 if abs(ema_x_err) < X_DEADBAND else ema_x_err

            # PID steering to keep the target centered
            now = time.time()
            dt  = now - last_t
            last_t = now
            turn = pid_turn.update(x_err, dt)
            turn = max(-STEER_LIMIT, min(STEER_LIMIT, turn))

            # Forward/back remains proportional to distance error
            fwd  = K_drive * r_err + (FWD_BIAS if r_err > 0 else 0.0)

            # Optional: de-emphasize forward motion when far off-center
            offset_factor = max(0.2, 1.0 - abs(x_err) / (W // 2))
            fwd *= offset_factor

            # Mix to differential drive
            left_cmd  = max(-MAX_SPEED, min(MAX_SPEED,  fwd - turn))
            right_cmd = max(-MAX_SPEED, min(MAX_SPEED,  fwd + turn))
            drive(left_cmd, right_cmd)

            cv2.putText(display, f"x:{x_err} r:{r_err} L:{left_cmd:.2f} R:{right_cmd:.2f}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            # If target lost: coast briefly, then stop & reset PID state
            if time.time() - last_seen_t > NO_TARGET_BRAKE_SEC:
                stop()
                pid_turn.reset()
                ema_x_err = None
            cv2.putText(display, "No target", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Show windows (Camera window removed)
        # cv2.imshow("Camera", frame)  # <-- removed as requested
        cv2.imshow("Mask", mask)
        cv2.imshow("Detected Colour", imgColor)
        cv2.imshow("Tracking+Control", display)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
finally:
    stop()
    cap.release()
    cv2.destroyAllWindows()
