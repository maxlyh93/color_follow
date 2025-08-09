#!/usr/bin/env python3
# Optimized color-following bot for Raspberry Pi 4 + USB webcam
# - Lower-res MJPG capture
# - ROI search after lock
# - Frame skipping + throttled control loop
# - Single window (optional masks), slower GUI refresh
# - Minimal per-frame allocations

import cv2
import numpy as np
import time
from gpiozero import Motor

# ========= FEATURE FLAGS / RUNTIME TUNING =========
USE_TRACKBARS   = True    # set False once HSV is tuned to save CPU
SHOW_MASKS      = False   # set True temporarily if you want to visualize masks
MIRROR_FLIP     = False   # set True if you prefer mirrored camera view

# Display cadence
SHOW_EVERY = 3            # update GUI every N processed frames
SKIP       = 1            # process every (SKIP+1)th frame (1 => every 2nd frame)
FULL_EVERY = 15           # full-frame search every N processed frames (otherwise ROI)
CTRL_HZ    = 12           # Hz for PID/motor updates (throttled)

# ========= CAMERA SETUP (Pi 4 + UVC webcam friendly) =========
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # prefer V4L2 backend on Linux
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  424)   # 320x240 or 424x240 are good starts
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # keep only 1 buffered frame
cv2.setNumThreads(1)                      # avoid CPU oversubscription on Pi

# ========= HSV TRACKBARS (only when tuning) =========
def empty(a): pass

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
    if not USE_TRACKBARS:
        # Default range (tweak to your target color); example is bright green
        return [35, 80, 60], [85, 255, 255]
    hmin = cv2.getTrackbarPos("Hue Min", "TrackBars")
    smin = cv2.getTrackbarPos("Sat Min", "TrackBars")
    vmin = cv2.getTrackbarPos("Val Min", "TrackBars")
    hmax = cv2.getTrackbarPos("Hue Max", "TrackBars")
    smax = cv2.getTrackbarPos("Sat Max", "TrackBars")
    vmax = cv2.getTrackbarPos("Val Max", "TrackBars")
    return [hmin, smin, vmin], [hmax, smax, vmax]

if USE_TRACKBARS:
    initTrackbars()

# ========= IMAGE PROCESSING (reuse kernel; minimal allocations) =========
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

def findColor(img, lower, upper):
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    # If blobs are holey, you can add one close:
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
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

def crop_roi(frame, center, r, margin=60):
    H, W = frame.shape[:2]
    x, y = center
    x0 = max(0, x - r - margin); x1 = min(W, x + r + margin)
    y0 = max(0, y - r - margin); y1 = min(H, y + r + margin)
    return frame[y0:y1, x0:x1], (x0, y0)

# ========= MOTORS (pins per Oivio Pi) =========
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

# ========= CONTROL GAINS & LIMITS =========
# Distance/approach control (keep object at desired size)
K_drive        = 0.010      # radius error -> forward/back strength
FWD_BIAS       = 0.15       # base bias to overcome static friction
MAX_SPEED      = 0.40       # cap for stability
DESIRED_RADIUS = 60         # pixels (tune to your camera/height)
NO_TARGET_BRAKE_SEC = 0.5

# Centering helpers (PID steering)
X_DEADBAND   = 8            # pixels tolerated around center
STEER_LIMIT  = 0.60         # absolute cap for turn command
SMOOTH_ALPHA = 0.60         # EMA filter for x_err (0..1)

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

# ========= MAIN LOOP =========
last_seen_t = time.time()
last_t      = time.time()
last_ctrl_t = time.time()
ema_x_err   = None
last_center, last_radius = None, None
frame_i = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            # camera hiccup; keep loop alive briefly
            cv2.waitKey(1)
            continue

        if MIRROR_FLIP:
            frame = cv2.flip(frame, 1)

        # Frame skipping to reduce load
        if frame_i % (SKIP + 1) != 0:
            # Keep GUI responsive on skipped frames
            if USE_TRACKBARS or SHOW_MASKS:
                cv2.waitKey(1)
            frame_i += 1
            continue

        H, W = frame.shape[:2]
        cx_target = W // 2

        # HSV ranges
        lower, upper = getTrackbarValues()

        # ROI search after we have a lock; full scan periodically
        search_frame = frame
        offset = (0, 0)
        if last_center and (frame_i % FULL_EVERY != 0):
            r = last_radius or 40
            search_frame, (ox, oy) = crop_roi(frame, last_center, r, margin=60)
            offset = (ox, oy)

        mask, imgColor = findColor(search_frame, lower, upper)
        circle, center, contour = getLargestContour(mask, min_area=600)

        # Promote ROI results back to full-frame coordinates
        if center:
            center = (center[0] + offset[0], center[1] + offset[1])
            if circle:
                circle = (circle[0] + offset[0], circle[1] + offset[1], circle[2])
                last_radius = circle[2]
            last_center = center
            last_seen_t = time.time()

        # CONTROL: throttle motor updates to CTRL_HZ
        now = time.time()
        dt  = now - last_t
        last_t = now

        if center:
            # Draw directly on the frame (avoid .copy())
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0,0,255), -1)
            if circle:
                cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), (0,255,0), 2)

            # Errors
            x_err_raw = center[0] - cx_target             # + if target is to the right
            r_err     = DESIRED_RADIUS - (circle[2] or 0) # + if too far away

            # Smooth + deadband horizontal error
            ema_x_err = x_err_raw if ema_x_err is None else int(
                SMOOTH_ALPHA * ema_x_err + (1 - SMOOTH_ALPHA) * x_err_raw
            )
            x_err = 0 if abs(ema_x_err) < X_DEADBAND else ema_x_err

            # Throttled PID & drive
            if now - last_ctrl_t >= 1.0 / CTRL_HZ:
                last_ctrl_t = now
                turn = pid_turn.update(x_err, dt)
                turn = max(-STEER_LIMIT, min(STEER_LIMIT, turn))

                fwd  = K_drive * r_err + (FWD_BIAS if r_err > 0 else 0.0)
                offset_factor = max(0.2, 1.0 - abs(x_err) / (W // 2))
                fwd *= offset_factor

                left_cmd  = max(-MAX_SPEED, min(MAX_SPEED,  fwd - turn))
                right_cmd = max(-MAX_SPEED, min(MAX_SPEED,  fwd + turn))
                drive(left_cmd, right_cmd)

                # HUD text less frequently (ties to control rate)
                cv2.putText(frame, f"x:{x_err} r:{r_err} L:{left_cmd:.2f} R:{right_cmd:.2f}",
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        else:
            # Target lost: coast briefly, then stop & reset
            if time.time() - last_seen_t > NO_TARGET_BRAKE_SEC:
                stop()
                pid_turn.reset()
                ema_x_err = None
                last_center, last_radius = None, None
            cv2.putText(frame, "No target", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ======= GUI (single main window; others optional & slower cadence) =======
        if frame_i % SHOW_EVERY == 0:
            try:
                cv2.imshow("Tracking+Control", frame)
                if SHOW_MASKS:
                    cv2.imshow("Mask", mask)
                    cv2.imshow("Detected Colour", imgColor)
                if USE_TRACKBARS:
                    # keep trackbar window live
                    pass
            except cv2.error:
                # Display can fail if no X server; ignore in headless mode
                pass

        # key handling (single waitKey per loop)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            motors_enabled = not motors_enabled
            if not motors_enabled:
                stop()
            print(f"Motors {'ENABLED' if motors_enabled else 'DISABLED'}")

        frame_i += 1

finally:
    stop()
    cap.release()
    cv2.destroyAllWindows()
