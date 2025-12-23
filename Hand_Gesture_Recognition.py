import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


# -------------------------
# Config
# -------------------------
# General configuration parameters for the application.
# Includes camera index, contour area thresholds, and convexity defect criteria.
# The kernel is used for morphological operations to 
# clean up the skin-color mask.
CAM_INDEX = 0
MIN_CONTOUR_AREA = 2000
DEFECT_DEPTH_RATIO = 0.008
DEFECT_ANGLE_THRESH = 80
kernel = np.ones((5, 5), np.uint8)
cap = cv2.VideoCapture(CAM_INDEX)


# -------------------------
# Tkinter GUI Setup
# -------------------------
# Initializes the main application window using Tkinter.
# Creates a label to display the live camera feed.
# The layout is prepared to hold both video and control widgets.
root = tk.Tk()
root.title("Hand Gesture Recognition")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height - 100}")

# -------------------------
# Main Layout Frames
# -------------------------
video_frame = tk.Frame(root, bg="black")
video_frame.pack(side="top", fill="both", expand=True)

controls_frame = tk.Frame(root, bg="#222", height=85)
controls_frame.pack(side="bottom", fill="x")

spacer = tk.Frame(root, height=15, bg="black")
spacer.pack(side="top", fill="x")

video_label = tk.Label(video_frame)
video_label.pack(fill="both", expand=True)


# -------------------------
# HSV Sliders (Tkinter)
# -------------------------
# Interactive sliders allow real-time adjustment of HSV skin-color thresholds.
# Users can fine-tune H (hue), S (saturation), and V (value) ranges for better hand segmentation.
# Default values are set to typical skin-tone ranges under normal lighting.
h_low = tk.Scale(controls_frame, from_=0, to=179,
                 orient="horizontal", label="H Low",
                 length=250)
h_low.set(0)
h_low.pack(side="left", padx=5)

h_high = tk.Scale(controls_frame, from_=0, to=179,
                  orient="horizontal", label="H High",
                  length=250)
h_high.set(20)
h_high.pack(side="left", padx=5)

s_low = tk.Scale(controls_frame, from_=0, to=255,
                 orient="horizontal", label="S Low",
                 length=250)
s_low.set(30)
s_low.pack(side="left", padx=5)

s_high = tk.Scale(controls_frame, from_=0, to=255,
                  orient="horizontal", label="S High",
                  length=250)
s_high.set(150)
s_high.pack(side="left", padx=5)

v_low = tk.Scale(controls_frame, from_=0, to=255,
                 orient="horizontal", label="V Low",
                 length=250)
v_low.set(60)
v_low.pack(side="left", padx=5)

v_high = tk.Scale(controls_frame, from_=0, to=255,
                  orient="horizontal", label="V High",
                  length=250)
v_high.set(255)
v_high.pack(side="left", padx=5)


# -------------------------
# Angle Calculation
# -------------------------
# Computes the angle (in degrees) at the farthest point of a convexity defect.
# Used to distinguish real finger valleys from noise or minor contour irregularities.
# Uses the law of cosines on three points: two hull points and one defect point.
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
# Core loop that captures, processes, and displays each video frame in real time.
# Performs hand detection, contour analysis, and gesture classification.
# Updates the Tkinter GUI with annotated video and debug previews.
def update_frame():
    
    # -------------------------
    # Frame Capture & Validation
    # -------------------------
    # Reads a frame from the camera; skips processing if capture fails.
    # Ensures robustness against camera disconnection or driver issues.
    # Uses a non-blocking callback via root.after() to maintain GUI responsiveness.
    ret, frame = cap.read()
    if not ret or frame is None:
        root.after(30, update_frame)
        return
    
    # -------------------------
    # Frame Mirroring & Backup
    # -------------------------
    # Horizontally flips the frame to mimic a mirror (natural for gesture interaction).
    # Keeps an unmodified copy for debugging previews later in the pipeline.
    # Stores frame dimensions for ROI and overlay positioning.
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    video_height = screen_height - 200
    frame = cv2.resize(frame, (screen_width, video_height))

    h, w = frame.shape[:2]

    
    # -------------------------
    # Region of Interest (ROI) Setup
    # -------------------------
    # Defines a fixed region where hand detection is performed (reduces noise from background).
    # Draws a blue bounding box to visually indicate the active area to the user.
    # Crops the working frame to this region for efficient processing.
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
    # Converts the ROI to HSV color space for robust skin-tone segmentation.
    # Applies real-time HSV thresholds from user-adjustable sliders.
    # Generates a binary mask where white pixels correspond to potential skin regions.
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
    # Applies Gaussian blur to reduce noise in the mask.
    # Uses morphological opening to remove small specks and closing to fill holes.
    # Final dilation slightly expands the mask to reconnect fragmented hand regions.
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # -------------------------
    # Hand Contour Selection
    # -------------------------
    # Finds all external contours in the refined mask.
    # Selects the largest contour that meets area criteria (not too small, not the whole frame).
    # This contour is assumed to represent the user's hand if valid.
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
    # Computes the convex hull of the hand contour to find finger valleys (defects).
    # For each defect, checks angle and depth to filter out false positives.
    # Counts valid defects to estimate the number of extended fingers.
    fingers_defects = 0
    
    if hand_detected:
        hull_points = cv2.convexHull(max_contour)
        cv2.drawContours(frame, [hull_points + [x1, y1]], -1, (255, 255, 0), 2)
        
        hull_idx = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull_idx)
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
        # Classifies hand as "Open" if exactly 4 valid defects (i.e., 5 fingers) are detected.
        # Displays real-time feedback on hand state and estimated finger count.
        # Visual indicators (text and color) help the user understand system interpretation.
        hand_open = fingers_defects == 4
        fingers_counts = fingers_defects + 1 if fingers_defects >= 1 else 'One finger or none'
        
        final_state = "Open" if hand_open else "Closed"
        color_final = (0, 255, 0) if hand_open else (0, 0, 255)
        
        cv2.drawContours(frame, [max_contour + [x1, y1]], -1, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Hand: {final_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_final, 3)
        cv2.putText(frame, f"Fingers: {fingers_counts}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,165), 2)
    
    else:
        # -------------------------
        # No Hand Detected Handling
        # -------------------------
        # Displays a warning message when no valid hand contour is found.
        # Helps the user adjust position or lighting conditions.
        # Shows "None" for finger count to indicate absence of gesture input.
        cv2.putText(frame, "No Hand Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Fingers: None", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,165), 2)
        
    # -------------------------
    # Debug Previews (Mask & Original)
    # -------------------------
    # Embeds small previews of the original frame and binary mask in the bottom corners.
    # Useful for tuning HSV sliders and diagnosing segmentation issues.
    # Resizes previews to avoid obstructing the main view.
    try:
        small_w = w // 5
        small_h = int(h / 6)
        
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_small = cv2.resize(mask_bgr, (small_w, small_h))
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
    # Overlays a preprocessed image (e.g., filtered version of 'bahget.png') based on finger count.
    # Each gesture (1–5 fingers) triggers a different visual filter for interactive feedback.
    # The overlay is positioned in the top-right corner with a labeled border.
    if fingers_defects == 1:
        overlay = sepia
        filterr = 'Sepia'
    elif fingers_defects == 2:
        overlay = posterized
        filterr = 'Posterized'
    elif fingers_defects == 3:
        overlay = heatmap
        filterr = 'Heatmap'
    elif fingers_defects == 4:
        overlay = bahget_clear
        filterr = 'Clear'
    else:
        overlay = bahget_blurred
        filterr = 'Blurred'

    oh, ow = overlay.shape[:2]

    # Top-right corner (with small margin)
    x_start = w - ow - 10
    y_start = 10

    frame[y_start:y_start+oh, x_start:x_start+ow] = overlay

    cv2.rectangle(frame,
                (x_start, y_start),
                (x_start+ow, y_start+oh),
                (0, 255, 255), 2)

    cv2.putText(frame, f"Filter: {filterr}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (123,123,73), 2)



    # -------------------------
    # GUI Frame Update
    # -------------------------
    # Converts the processed OpenCV (BGR) frame to PIL-compatible RGB format.
    # Wraps it in a PhotoImage for display in the Tkinter label.
    # Schedules the next frame update after 30ms to maintain real-time interaction.
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im_pil)
    
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    root.after(30, update_frame)


# -------------------------
# Image Filters
# -------------------------
# Precomputes several stylized versions of a reference image ('bahget.png').
# Includes effects like blur, sepia, posterization, and heatmap for gesture-based switching.
# All filtered images are resized to a consistent dimension for seamless overlay.
bahget_clear = cv2.imread("bahget.png")
bahget_clear = cv2.resize(bahget_clear, (350, 250))

# Create blurred version (default state)
bahget_blurred = cv2.GaussianBlur(bahget_clear, (63, 63), 0)

# Create heatmap version
# COLORMAP_JET:
#   src => gray => (0 -> 255)
#   0 -> blue
#   128 -> green / yellow
#   255 -> red
heatmap = cv2.applyColorMap(cv2.cvtColor(bahget_clear, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

# Create sepia version
                        #   R      G      B
sepia_filter = np.array([[0.272, 0.534, 0.131],   # B
                         [0.349, 0.686, 0.168],   # G
                         [0.393, 0.769, 0.189]])  # R
sepia = cv2.transform(bahget_clear, sepia_filter)
sepia = np.clip(sepia, 0, 255).astype(np.uint8)

# Create posterized version
levels = 4
# 0 – 63   → 0
# 64 –127  → 64
# 128–191  → 128
# 192–255  → 192
posterized = np.floor_divide(bahget_clear, 256//levels) * (256//levels) # Quantization
posterized = posterized.astype(np.uint8)


# -------------------------
# Start Main Loop
# -------------------------
# Launches the Tkinter event loop and begins real-time video processing.
# The GUI remains responsive to slider adjustments and window events.
# This is the entry point for the interactive application.
update_frame()
root.mainloop()


# -------------------------
# Cleanup Resources
# -------------------------
# Releases the camera handle and closes any OpenCV-created windows.
# Ensures clean termination and prevents resource leaks after closing the app.
# Note: These lines may not execute if the user force-quits the GUI.
cap.release()
cv2.destroyAllWindows()