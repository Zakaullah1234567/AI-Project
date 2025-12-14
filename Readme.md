 TraffiSense – AI-Based Intelligent Traffic Light Control System

## Overview
TraffiSense is an **AI-powered intelligent traffic light control system** that:

- Detects vehicles in **real-time** using camera feeds
- Dynamically adjusts **traffic signal timing** based on vehicle density
- Gives **priority to emergency vehicles** (ambulances, fire trucks)

This system demonstrates how **computer vision + adaptive algorithms** can improve traffic flow and reduce congestion.

---

## Core Technologies Used

- **Python**
- **OpenCV** – Video processing
- **YOLOv8 (Ultralytics)** – Real-time vehicle detection
- **Multithreading** – Parallel camera processing
- **Queues** – Safe inter-thread communication

---

## File Structure & Imports

```python
import cv2           # OpenCV - for video/image processing
import numpy as np   # NumPy - for mathematical operations
import time          # Time - for delays and timestamps
import threading     # Threading - for running multiple tasks simultaneously
import queue         # Queue - for safe data sharing between threads
from pathlib import Path  # Path - for file path operations
from ultralytics import YOLO  # YOLO - AI model for object detection
```

---

## Configuration Section
Customize the system behavior from this section:

```python
# Timing settings
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 40
YELLOW_TIME = 3

# AI detection settings
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5

# Vehicle classes
EMERGENCY_CLASSES = ['ambulance', 'fire truck']
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'Bike', 'bicycle'] + EMERGENCY_CLASSES

# Video sources
VIDEO_SOURCES = {
    'north': r'D:\Hello\TraffiSense\videos\traffic_north.mp4',
    'south': r'D:\Hello\TraffiSense\videos\traffic_south.mp4',
}
```

---

## Class-by-Class Explanation

### 1 TrafficSignal Class
**Purpose:** Represents a single traffic light (North, South, East, or West)

- Stores light state (RED / YELLOW / GREEN)
- Tracks vehicle count
- Handles emergency status

> Think of it as a controller for **one road direction**.

---

### 2 VehicleDetector Class
**Purpose:** Uses YOLOv8 AI model to detect vehicles

**Key Features:**
- Loads YOLOv8 model
- Detects vehicles in each frame
- Draws bounding boxes
- Identifies emergency vehicles

**Main Method:**
```python
detect_vehicles(frame)
```

**Input:** Video frame  
**Output:** Detected vehicles with confidence scores

---

### 3 TrafficCamera Class
**Purpose:** Manages one camera feed

**Responsibilities:**
- Reads frames from video/webcam
- Sends frames to AI detector
- Counts vehicles
- Displays processed video

**Modes Supported:**
- Real video mode
- Simulation mode (fake traffic generation)

---

### 4 TrafficController Class (Main Brain)
The most important component of the system.

#### A. Adaptive Timing Logic
```text
Green Time = Base Time + (Vehicle Count × 2 seconds)
```

- Minimum: 10 seconds
- Maximum: 40 seconds

Example:
> 5 vehicles → 10 + (5 × 2) = **20 seconds**

---

#### B. Emergency Priority System

**Flow:**
- Detect ambulance/fire truck
- Instantly give GREEN to that direction
- All others turn RED
- Hold green for 15 seconds
- Resume normal cycle

---

#### C. Traffic Light Cycle

```text
1. North-South GREEN
2. North-South YELLOW
3. North-South RED
4. East-West GREEN
5. East-West YELLOW
6. East-West RED
7. Repeat
```

---

#### D. Display System

- Console display (text logs)
- Video display (2×2 grid with detection boxes)

Color Codes:
- Green boxes → Normal vehicles
- Red boxes → Emergency vehicles

---

## How the System Works (Step-by-Step)

### Step 1: Initialization
- Load YOLOv8 model
- Create camera objects (N, S, E, W)
- Create traffic signals
- Start controller

### Step 2: Camera Threads
Each camera runs independently:

```text
Read Frame → Detect Vehicles → Count → Update Queue
```

### Step 3: Main Control Loop

```python
while system_running:
    get vehicle counts
    check emergency
    adjust timing
    update lights
    refresh display
```

---

## Visual Workflow

```text
Video Feeds → Camera Threads → Vehicle Queue → Main Controller
                                   ↓
                         Traffic Lights + Displays
```

---

## Key Features

### Multi-threading
- Each camera runs independently
- Faster and smoother processing

### Queue System
- Thread-safe data sharing
- Prevents data corruption

### Adaptive Algorithm
- More vehicles → Longer green time
- Fewer vehicles → Shorter green time

### Emergency Override
- Immediate green for emergency direction
- Life-saving priority handling

### Dual Display System
- Console logs
- Visual AI detection

---

## How to Run the System

### Option 1: Real Video Files
```bash
python traffisense.py
```
Ensure video paths are correct.

### Option 2: Simulation Mode
```python
USE_SIMULATION = True
```

### Option 3: Webcam Mode
```python
VIDEO_SOURCES = {
    'north': 0,
    'south': 1
}
```

---

## Troubleshooting

| Issue | Solution |
|------|---------|
| No video found | Enable simulation mode |
| YOLO not downloading | Install ultralytics manually |
| No video window | Ensure OpenCV is installed |
| Messy console | Normal (updates every second) |

---

## Learning Opportunities

### For Beginners
- Modify configuration values
- Observe timing changes
- Add print statements

### To Extend
- Add new vehicle classes
- Modify timing algorithm
- Add pedestrian signals

### To Test
- Use simulation mode
- Trigger emergency scenarios

---

## Developer

**Developed by:** Zaka Ullah  
**Project Name:** TraffiSense  
**Domain:** AI | Computer Vision | Smart Cities

---

*If you find this project useful, please give it a star on GitHub!*

