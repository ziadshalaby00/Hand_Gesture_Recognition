# Vision-Based Human-Robot Interaction System

A real-time hand gesture recognition system that controls a 3D humanoid robot through computer vision. The system captures hand landmarks via webcam, interprets gestures into robot commands, and streams both the camera feed and 3D animation to a web dashboard.

---

## How It Works

The pipeline captures live video, detects hand landmarks using **MediaPipe**, maps finger counts and palm position to robot actions, then broadcasts the data to a web interface where a 3D character performs the corresponding animation in real time.

---

## Gesture → Action Mapping

| Fingers | Action |
|---------:|--------|
| 0 | Sit |
| 1 | Walk |
| 2 | Run |
| 3 | Dance |
| 4 | Punch |
| 5 + Move Left/Right | Strafe |

---

## Tech Stack

### Computer Vision & Backend

- **MediaPipe** — Hand landmark detection
- **OpenCV** — Video capture and frame processing
- **FastAPI + Uvicorn** — MJPEG video stream server
- **Python-SocketIO** — Real-time telemetry broadcasting

### Frontend & 3D

- **Three.js** — WebGL 3D viewport and character rendering
- **GLTF/GLB** — Animated robot model and motion clips
- **Socket.IO Client** — Live data ingestion
- **CSS3** — Glassmorphism UI and responsive layout

---

## Architecture

```text
┌─────────────┐        ┌──────────────┐        ┌─────────────────┐
│   Webcam    │ ──────▶   core.py     ───────▶│  video_server   │
│  (OpenCV)   │        │ (MediaPipe)  │        │ (FastAPI :5000) │────────────┐
│             │        │ (Processing) │        │ (MJPEG Stream)  │            │
└─────────────┘        └──────┬───────┘        └─────────────────┘            │
                              │                                               │
                              ▼                                               │
                     ┌──────────────┐                                         │
                     │  robot_3d.py │──────▶ Socket.IO Server (:8765)         │
                     │ (Socket.IO   │                 │                       │
                     │   Client)    │                 │                       │
                     └──────────────┘                 ▼                       │
                                                ┌─────────────┐               │
                                                │ robot-web/  │ ◀─────────────┘
                                                │ Three.js UI │
                                                └─────────────┘
```

---

## Quick Start

```bash
# 1. Setup environment
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt

# 2. Start the Socket.IO server
python socket_io_server.py

# 3. Start the vision pipeline
python core.py

# 4. Serve robot-web/index.html via Live Server

# 5. Open the browser and wait for initialization
```

> **Note:** Both backend services must be running before refreshing the browser. A brief startup delay is normal while the MediaPipe model initializes.

---

## File Overview

| File | Purpose |
|------|---------|
| `core.py` | Main vision pipeline — detection, gesture logic, frame dispatch |
| `video_server.py` | FastAPI MJPEG streaming endpoint |
| `socket_io_server.py` | Socket.IO hub for real-time telemetry |
| `robot_3d.py` | Socket.IO client bridge that pushes hand data |
| `robot-web/` | Browser-based 3D dashboard |
| `Temp.py` | Standalone Tkinter testing utility |

---

## 📄 License

Developed entirely by [Ziad Shalaby](https://github.com/ziadshalaby00).

This project is licensed under the **MIT License**.

---
