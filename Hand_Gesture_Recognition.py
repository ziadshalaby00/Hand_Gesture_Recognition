import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


# -------------------------
# Config
# -------------------------
# Core constants that control detection sensitivity and behavior.
# MIN_CONTOUR_AREA filters out noise by ignoring tiny blobs.
# DEFECT_DEPTH_RATIO and DEFECT_ANGLE_THRESH jointly validate finger gaps:
# depth must be proportional to hand size, and the valley must be sharp enough.
# The 5x5 kernel is used in morphological operations to clean the skin mask.
CAM_INDEX = 0
MIN_CONTOUR_AREA = 2000
DEFECT_DEPTH_RATIO = 0.008
DEFECT_ANGLE_THRESH = 80
kernel = np.ones((5, 5), np.uint8)
cap = cv2.VideoCapture(CAM_INDEX)


# -------------------------
# Tkinter GUI Setup
# -------------------------
# Basic GUI window with a label to display the live video feed.
# The grid layout reserves space for both video and control sliders.
root = tk.Tk()
root.title("Hand Gesture Recognition")

screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.geometry(f"{screen_w}x{screen_h}+0+0")

try:
    root.state("zoomed")
except:
    pass

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

video_label = tk.Label(root)
video_label.grid(row=0, column=0, columnspan=6, sticky="nsew")


# -------------------------
# HSV Sliders (Tkinter)
# -------------------------
# Interactive HSV threshold sliders for real-time skin-color tuning.
# OpenCV uses H∈[0,179], S/V∈[0,255]. Default values target typical light-to-olive skin tones.
# These allow on-the-fly calibration under varying lighting conditions.
h_low = tk.Scale(root, from_=0, to=179, orient="horizontal", label="H Low")
h_low.set(0)
h_low.grid(row=1, column=0, sticky="ew")

h_high = tk.Scale(root, from_=0, to=179, orient="horizontal", label="H High")
h_high.set(20)
h_high.grid(row=1, column=1, sticky="ew")

s_low = tk.Scale(root, from_=0, to=255, orient="horizontal", label="S Low")
s_low.set(30)
s_low.grid(row=1, column=2, sticky="ew")

s_high = tk.Scale(root, from_=0, to=255, orient="horizontal", label="S High")
s_high.set(150)
s_high.grid(row=1, column=3, sticky="ew")

v_low = tk.Scale(root, from_=0, to=255, orient="horizontal", label="V Low")
v_low.set(60)
v_low.grid(row=1, column=4, sticky="ew")

v_high = tk.Scale(root, from_=0, to=255, orient="horizontal", label="V High")
v_high.set(255)
v_high.grid(row=1, column=5, sticky="ew")


# -------------------------
# Helper Function: Angle Calculation
# -------------------------
# Computes the interior angle (in degrees) at point p2 formed by points (p1, p2, p3).
# Uses the law of cosines with numerical safeguards:
# - Handles degenerate cases (zero-length edges) by returning 180°.
# - Clips cosine values to [-1, 1] to prevent NaN from arccos due to floating-point errors.
def angle_between(p1, p2, p3):
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p1 - p3)
    
    if a * b == 0:
        return 180
    
    cos_angle = (a * a + b * b - c * c) / (2 * a * b)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    return np.degrees(np.arccos(cos_angle))


# -------------------------
# Main Frame Update Function
# -------------------------
# The core real-time processing loop:
# 1. Captures and mirrors the frame for natural interaction.
# 2. Defines a fixed ROI (top-left) to reduce computation and avoid background interference.
# 3. Converts to HSV and applies dynamic skin-color masking using slider values.
# 4. Refines the mask using Gaussian blur and morphological operations (open/close/dilate)
#    to remove noise and fill holes while preserving hand shape.
# 5. Finds the largest valid contour (hand candidate) based on area constraints.
# 6. If a hand is found, computes its convex hull and convexity defects.
#    Each defect is validated by depth (relative to hand area) and tip angle—
#    only sharp, deep valleys are counted as finger gaps.
# 7. Classifies hand as "Open" if ≥2 valid defects (typically 4 fingers → 3–4 defects).
# 8. Triggers action_on() or action_off() based on state.
# 9. Overlays debugging visuals: hull, defect points, state label.
# 10. Adds small previews (original + mask) at the bottom for tuning feedback.
# 11. Converts frame to PIL format and updates Tkinter display.
# Runs every ~30ms via root.after() for smooth video.
def update_frame():
    
    # -------------------------
    # Frame Capture & Validation
    # -------------------------
    # Grab a new frame from the camera. If capture fails (e.g., camera disconnected),
    # skip processing and reschedule the next update to keep the loop alive.
    ret, frame = cap.read()
    if not ret or frame is None:
        root.after(30, update_frame)
        return
    
    # -------------------------
    # Frame Mirroring & Backup
    # -------------------------
    # Mirror the frame horizontally for natural user interaction (like a mirror).
    # Keep a copy of the original for the small preview thumbnail later.
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    # -------------------------
    # Region of Interest (ROI) Setup
    # -------------------------
    # Define a fixed top-left ROI to limit processing area, reduce noise,
    # and avoid interference from background objects. Draw a blue rectangle
    # to visually indicate the active zone to the user.
    ROI_W = (w // 2)
    ROI_H = (h * 4) // 5
    x1 = 10
    y1 = 10
    x2 = x1 + ROI_W
    y2 = y1 + ROI_H
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    work_frame = frame[y1:y2, x1:x2]
    
    # -------------------------
    # Dynamic Skin Color Masking (HSV)
    # -------------------------
    # Convert ROI to HSV color space and apply a binary mask using live
    # slider values. This allows real-time tuning of skin detection under
    # different lighting or skin tones without restarting the app.
    hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
    
    hL = h_low.get()
    hH = h_high.get()
    sL = s_low.get()
    sH = s_high.get()
    vL = v_low.get()
    vH = v_high.get()

    mask = cv2.inRange(hsv, (hL, sL, vL), (hH, sH, vH))
    
    # -------------------------
    # Mask Refinement Pipeline
    # -------------------------
    # Clean the binary mask in stages:
    # 1. Gaussian blur to soften edges and reduce sensor noise.
    # 2. Morphological opening to remove small specks.
    # 3. Closing to fill holes inside the hand region.
    # 4. Final dilation to reconnect slightly broken contours.
    # This pipeline ensures a solid, noise-free hand silhouette.
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # -------------------------
    # Hand Contour Selection
    # -------------------------
    # Find all external contours and sort by area (largest first).
    # Pick the first contour that is:
    # - Large enough to be a hand (≥ MIN_CONTOUR_AREA)
    # - Not too large (avoids full-background false positives)
    # This balances robustness and precision in detection.
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
    # Hand Geometry Analysis (If Detected)
    # -------------------------
    # When a hand is found:
    # - Compute its convex hull (outer boundary).
    # - Calculate convexity defects (indentations between fingers).
    # - For each defect, validate using two criteria:
    #     * Depth > DEFECT_DEPTH_RATIO × hand area → ensures it's significant.
    #     * Tip angle < DEFECT_ANGLE_THRESH → ensures it's sharp (not a curve).
    # Valid defects are counted as finger gaps. Mark them with red dots.
    if hand_detected:
        hull_points = cv2.convexHull(max_contour)
        cv2.drawContours(frame, [hull_points + [x1, y1]], -1, (255, 255, 0), 2)
        
        hull_idx = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull_idx)
        
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
        # Hand State Classification
        # -------------------------
        # Heuristic: ≥2 valid defects → "Open hand" (typically 3–4 for spread fingers).
        # Fewer → "Closed" (fist or partial grip). Trigger corresponding actions
        # and overlay a colored status label on the main feed.
        hand_open = fingers_defects >= 2
        final_state = "Open" if hand_open else "Closed"
        color_final = (0, 255, 0) if hand_open else (0, 0, 255)
        
        action_on() if hand_open else action_off()
        
        cv2.drawContours(frame, [max_contour + [x1, y1]], -1, (255, 0, 0), 2)
        cv2.putText(frame, f"Hand: {final_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_final, 3)
    
    else:
        # -------------------------
        # No Hand Detected Handling
        # -------------------------
        # Display a warning message and ensure the "off" action is triggered.
        # This provides clear feedback and prevents stale state if the hand leaves ROI.
        cv2.putText(frame, "No Hand Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        action_off()
        
    # -------------------------
    # Debug Previews (Mask & Original)
    # -------------------------
    # Embed two small thumbnails at the bottom of the main frame:
    # - Left: processed skin mask (for tuning HSV sliders)
    # - Right: original unprocessed frame
    # Enclosed in white borders for visibility. Wrapped in try/except
    # to prevent crashes if frame dimensions change unexpectedly.
    try:
        small_w = w // 5
        small_h = int(h / 6)
        
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_rgb, (small_w, small_h))
        frame_small = cv2.resize(frame_copy, (small_w, small_h))
        
        y1 = h - small_h - 7
        y2 = h - 7
        
        x1_left = 10
        x2_left = x1_left + small_w
        frame[y1:y2, x1_left:x2_left] = mask_small
        cv2.rectangle(frame, (x1_left, y1), (x2_left, y2), (255, 255, 255), 1)
        
        x1_right = w - small_w - 10
        x2_right = w - 10
        frame[y1:y2, x1_right:x2_right] = frame_small
        cv2.rectangle(frame, (x1_right, y1), (x2_right, y2), (255, 255, 255), 1)
    
    except:
        pass
    
    # -------------------------
    # Action Image (Top-Right Corner)
    # -------------------------
    if action_active:
        overlay = bahget_clear     # hand open → reveal
    else:
        overlay = bahget_blurred   # hand closed → blurred

    oh, ow = overlay.shape[:2]

    # Top-right corner (with small margin)
    x_start = w - ow - 10
    y_start = 10

    frame[y_start:y_start+oh, x_start:x_start+ow] = overlay

    cv2.rectangle(frame,
                (x_start, y_start),
                (x_start+ow, y_start+oh),
                (0, 255, 255), 2)



    # -------------------------
    # GUI Frame Update
    # -------------------------
    # Convert the final OpenCV BGR frame to RGB, then to a PIL ImageTk object
    # compatible with Tkinter. Update the label and schedule the next frame
    # after ~30ms (~33 FPS) to maintain smooth real-time video.
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    root.after(30, update_frame)


# -------------------------
# Load Action Image
# -------------------------
bahget_clear = cv2.imread("bahget.png")
bahget_clear = cv2.resize(bahget_clear, (350, 250))

# Create blurred version (default state)
bahget_blurred = cv2.GaussianBlur(bahget_clear, (31, 31), 0)

action_active = False


# -------------------------
# Action On/Off
# -------------------------
# Placeholder functions for custom logic (e.g., triggering LED, sending signal).
# Override these to integrate with hardware or other systems.
def action_on():
    global action_active
    action_active = True


def action_off():
    global action_active
    action_active = False


# -------------------------
# Start Main Loop
# -------------------------
# Launches the real-time processing loop and GUI event handling.
# update_frame() is called once to start the cycle; root.mainloop() keeps the window alive.
update_frame()
root.mainloop()


# -------------------------
# Cleanup Resources
# -------------------------
# Releases camera and closes any OpenCV windows after GUI exits.
cap.release()
cv2.destroyAllWindows()