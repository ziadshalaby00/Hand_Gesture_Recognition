# Vision-Based Human-Robot Interaction System Using Hand Gestures

## Basic files
- socket_io_server.py
- video_server.py
- robot_3d.py
- core.py
- robot-web/
- - index.html 
- - style.css 
- - main.js 


## Project Setup & Run Guide

### 1. Create Virtual Environment

First, create a Python virtual environment:

```bash
python -m venv venv
```

---

### 2. Install Dependencies

Activate the virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

> Make sure the file name is exactly `requirements.txt`.

---

### 3. Open the Web Interface

Navigate to the project folder and open:

```
robot-web/index.html
```

Open this file with live server in vs code or any server like node js.

---

### 4. Run Backend Services

Open **two separate terminals**:

#### Terminal 1 — Socket.IO Server

Activate the virtual environment, then run:

```bash
py socket_io_server.py
```

---

#### Terminal 2 — Core Service

```bash
python core.py
```

---

### 5. Final Step

* Refresh the web page (`index.html`)
* Wait a few seconds for initialization
* Any initial lag or freezing should disappear after the system stabilizes

---

### Notes

* Both backend servers must be running before refreshing the browser
* A short delay or slight lag at startup is normal until everything connects

---

### License

MIT License © 2026 Ziad Ahmed Shalaby

