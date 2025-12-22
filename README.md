# Hand Gesture Recognition

Real-time **hand gesture recognition** using OpenCV, HSV color segmentation, convexity defects, and a Tkinter GUI.

---

## 📌 Overview

This project detects **Open vs Closed hand** gestures in real-time using:

* **HSV Skin Detection** with user-adjustable sliders
* **Morphological Filtering** for clean segmentation
* **Contours + Convex Hull**
* **Convexity Defects** to count fingers
* **Tkinter GUI** for live video display with overlays

All processing is done inside a defined **ROI** to reduce noise and improve stability.

---

## 🧩 Features

* Real-time gesture detection (Open / Closed hand)
* No machine learning required (lightweight & fast)
* Tkinter GUI with live feed and interactive sliders
* Accurate finger detection using convexity defects
* Custom visual feedback based on number of fingers (overlays, filters)

---

## 1. Camera Setup & Frame Capture

The camera is opened via:

```python
cap = cv2.VideoCapture(CAM_INDEX)
```

Frames are captured every ~30ms using:

```python
update_frame()
```

With safety check:

```python
if not ret or frame is None:
    return
```

---

## 2. ROI (Region of Interest)

A fixed ROI is defined to improve detection:

* **Width**: half of the frame
* **Height**: 4/5 of the frame

```python
work_frame = frame[y1:y2, x1:x2]
```

The frame is also **mirrored** for a natural interactive experience:

```python
frame = cv2.flip(frame, 1)
```

---

## 3. Convert Frame to HSV

HSV is used because it separates lightness from color, improving skin detection robustness:

```python
hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
```

---

## 4. Dynamic Skin Detection via HSV Sliders

Users can adjust sliders in real-time to fine-tune skin segmentation:

```python
mask = cv2.inRange(hsv, (hL, sL, vL), (hH, sH, vH))
```

* **White** = skin
* **Black** = background

Default slider values approximate normal skin tone under standard lighting.

---

## 5. Mask Cleaning (Noise Removal)

The mask is refined using a pipeline of filters:

```python
mask = cv2.GaussianBlur(mask, (7, 7), 0)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=1)
```

---

## 6. Contour Extraction

Hand candidates are extracted from the mask:

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

The **largest valid contour** within area thresholds is considered the hand.

---

## 7. Convex Hull

A convex hull is drawn around the hand contour:

```python
hull_points = cv2.convexHull(max_contour)
```

This provides a tight outer boundary for defect analysis.

---

## 8. Convexity Defects

Convexity defects represent the **valleys between fingers**:

```python
hull_idx = cv2.convexHull(max_contour, returnPoints=False)
defects = cv2.convexityDefects(max_contour, hull_idx)
```

Each defect is analyzed for:

* **Angle** (finger-like)
* **Depth** (significant enough relative to hand area)

```python
if ang < DEFECT_ANGLE_THRESH and depth > area * DEFECT_DEPTH_RATIO:
    fingers_defects += 1
```

---

## 9. Hand State Detection (Open/Closed)

The hand is classified as **Open** if **exactly 4 defects** are detected (5 fingers extended):

```python
hand_open = fingers_defects == 4
```

* Open → 5 fingers extended
* Closed → fewer than 5 fingers

Status and finger count are displayed on the GUI:

```python
cv2.putText(frame, f"Hand: {final_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_final, 3)
cv2.putText(frame, f"Fingers: {fingers_counts}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,165), 2)
```

---

## 10. Image Filters / Overlay

Custom overlays are shown in the top-right corner based on the number of fingers:

| Fingers     | Filter / Overlay |
| ----------- | ---------------- |
| 2           | Sepia            |
| 3           | Posterized       |
| 4           | Heatmap          |
| 5 / Open    | Clear            |
| 0 or others | Blurred          |

```python
frame[y_start:y_start+oh, x_start:x_start+ow] = overlay
```

A label is displayed:

```python
cv2.putText(frame, f"Filter: {filterr}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (123,123,73), 2)
```

---

## 11. GUI (Tkinter)

Tkinter is used to display frames in real-time:

```python
im_pil = Image.fromarray(img_rgb)
imgtk = ImageTk.PhotoImage(image=im_pil)
video_label.configure(image=imgtk)
```

Frames are updated every 30ms:

```python
root.after(30, update_frame)
```

---

## 📎 Additional Notes

* Optional debug previews show the mask and original frame in small corners.
* Sliders allow real-time tuning for different lighting conditions.
* Code is modular and extendable for custom actions on gestures.

---

## 🏁 How to Run

```bash
python Hand_Gesture_Recognition.py
```

Make sure the reference image (`bahget.png`) is in the same directory.

---

## 📜 License

Ziad Ahmed Shalaby License
