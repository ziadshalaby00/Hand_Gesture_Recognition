import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


# -------------------------
# Config
# -------------------------
CAM_INDEX = 0
MIN_CONTOUR_AREA = 2000
DEFECT_DEPTH_RATIO = 0.008
DEFECT_ANGLE_THRESH = 80
kernel = np.ones((5, 5), np.uint8)
cap = cv2.VideoCapture(CAM_INDEX)


# -------------------------
# Tkinter GUI Setup
# -------------------------
root = tk.Tk()
root.title("Hand Gesture Recognition")
video_label = tk.Label(root)
video_label.grid(row=0, column=0, columnspan=6)


# -------------------------
# Helper Function: Angle Calculation
# -------------------------
def angle_between(p1, p2, p3):
    # -------------------------
    # Compute side lengths of triangle
    # -------------------------
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p1 - p3)
    
    # -------------------------
    # Handle degenerate case
    # -------------------------
    if a * b == 0:
        return 180
    
    # -------------------------
    # Apply law of cosines
    # -------------------------
    cos_angle = (a * a + b * b - c * c) / (2 * a * b)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    # -------------------------
    # Convert to degrees
    # -------------------------
    return np.degrees(np.arccos(cos_angle))


# -------------------------
# Main Frame Update Function
# -------------------------
def update_frame():
    # -------------------------
    # Capture and validate frame
    # -------------------------
    ret, frame = cap.read()
    if not ret or frame is None:
        root.after(30, update_frame)
        return
    
    # -------------------------
    # Mirror and backup frame
    # -------------------------
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # -------------------------
    # Define and draw ROI
    # -------------------------
    ROI_W = w // 2
    ROI_H = (h * 4) // 5
    x1 = 10
    y1 = 10
    x2 = x1 + ROI_W
    y2 = y1 + ROI_H
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    work_frame = frame[y1:y2, x1:x2]
    
    # -------------------------
    # Skin detection via HSV
    # -------------------------
    hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 60), (20, 150, 255))
    
    # -------------------------
    # Noise removal and mask refinement
    # -------------------------
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # -------------------------
    # Find and sort contours
    # -------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_detected = False
    hand_open = False
    max_contour = None
    area = 0
    h_f, w_f = work_frame.shape[:2]
    
    if len(contours) > 0:
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours_sorted:
            area = cv2.contourArea(cnt)
            if area >= MIN_CONTOUR_AREA and area < h_f * w_f * 0.5:
                max_contour = cnt
                hand_detected = True
                break
    
    # -------------------------
    # Hand analysis if detected
    # -------------------------
    if hand_detected:
        # -------------------------
        # Draw convex hull
        # -------------------------
        hull_points = cv2.convexHull(max_contour)
        cv2.drawContours(frame, [hull_points + [x1, y1]], -1, (0, 255, 255), 2)
        
        # -------------------------
        # Compute convexity defects
        # -------------------------
        hull_idx = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull_idx)
        
        # -------------------------
        # Count valid finger defects
        # -------------------------
        fingers_defects = 0
        if defects is not None:
            for d in defects[:, 0]:
                s, e, f, depth = d
                start = max_contour[s][0]
                end = max_contour[e][0]
                far = max_contour[f][0]
                ang = angle_between(start, far, end)
                if ang < DEFECT_ANGLE_THRESH and depth > area * DEFECT_DEPTH_RATIO:
                    fingers_defects += 1
                    far_x = far[0] + x1
                    far_y = far[1] + y1
                    cv2.circle(frame, (far_x, far_y), 5, (0, 0, 255), -1)
        
        # -------------------------
        # Determine hand state
        # -------------------------
        hand_open = fingers_defects >= 2
        final_state = "Open" if hand_open else "Closed"
        color_final = (0, 255, 0) if hand_open else (0, 0, 255)
        
        # Start Action
        action_on(frame) if hand_open else action_off(frame)
        
        # -------------------------
        # Draw contour and label
        # -------------------------
        cv2.drawContours(frame, [max_contour + [x1, y1]], -1, (255, 0, 0), 2)
        cv2.putText(frame, f"Hand: {final_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_final, 3)
    
    else:
        # -------------------------
        # No hand detected message
        # -------------------------
        cv2.putText(frame, "No Hand Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Close Action
        action_off(frame)
        
    # -------------------------
    # Small preview windows (mask & original)
    # -------------------------
    try:
        small_w = w // 5
        small_h = int(h / 6)
        
        # -------------------------
        # Prepare small previews
        # -------------------------
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_rgb, (small_w, small_h))
        frame_small = cv2.resize(frame_copy, (small_w, small_h))
        
        # -------------------------
        # Position previews at bottom
        # -------------------------
        y1 = h - small_h - 7
        y2 = h - 7
        
        # Left preview (mask)
        x1_left = 10
        x2_left = x1_left + small_w
        frame[y1:y2, x1_left:x2_left] = mask_small
        cv2.rectangle(frame, (x1_left, y1), (x2_left, y2), (255, 255, 255), 1)
        
        # Right preview (original frame)
        x1_right = w - small_w - 10
        x2_right = w - 10
        frame[y1:y2, x1_right:x2_right] = frame_small
        cv2.rectangle(frame, (x1_right, y1), (x2_right, y2), (255, 255, 255), 1)
    
    except:
        pass
    
    # -------------------------
    # Convert to PIL and update Tkinter label
    # -------------------------
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    # -------------------------
    # Schedule next frame
    # -------------------------
    root.after(30, update_frame)


# -------------------------
# Action On/Off
# -------------------------
def action_on():
    pass

def action_off():
    pass


# -------------------------
# Start Main Loop
# -------------------------
update_frame()
root.mainloop()


# -------------------------
# Cleanup Resources
# -------------------------
cap.release()
cv2.destroyAllWindows()