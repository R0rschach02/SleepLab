# 💤 Sleep Motion Tracker (Proof of Concept)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📘 Overview

The **Sleep Motion Tracker (PoC)** is a lightweight prototype designed to detect and quantify movement during sleep using computer vision.  
It establishes a foundation for a future modular sleep analysis system that combines **video-based motion tracking** with **sensor data**.

The project uses an **infrared (IR) night-vision camera** and **OpenCV** to track motion in low-light environments and log data automatically for later analysis.

---

## 🧩 Features

- 📷 **IR-based motion detection** in dark environments  
- 🧠 **Adaptive background modeling** using MOG2  
- 🔧 **Noise filtering** with Gaussian blur & morphological operations  
- 📈 **CSV logging** of timestamps, motion states, and motion area  
- 🐍 Fully written in **Python** (runs on Raspberry Pi or desktop)

---

## ⚙️ Hardware & Software

| Component | Description |
| :--- | :--- |
| **Hardware** | Raspberry Pi 4, CSI night-vision (IR) camera |
| **OS** | Raspberry Pi OS / Ubuntu |
| **Language** | Python 3.10+ |
| **Libraries** | OpenCV, NumPy, Pandas (for log handling) |

---

## 🧠 Technical Overview

The project implements a robust motion-detection pipeline optimized for low-light environments.

1. **Camera Initialization**  
   Accesses video feed using `cv2.VideoCapture` (compatible with CSI or USB cameras).

2. **Background Modeling**  
   Uses `cv2.createBackgroundSubtractorMOG2` to build a dynamic model of the background — adapting to gradual lighting changes.

3. **Noise Reduction**  
   - `cv2.GaussianBlur` smooths pixel noise from the IR camera.  
   - Morphological operations (`cv2.morphologyEx`) remove small artifacts and stabilize motion contours.

4. **Motion Detection**  
   - Extracts contours using `cv2.findContours`.  
   - Filters small movements with a configurable `MIN_CONTOUR_AREA` threshold.

5. **Data Logging**  
   Logs timestamp, motion state (`True` / `False`), and motion area to `motion_log.csv` for later visualization.

---

## 💻 Installation

Clone the repository:
```bash
git clone https://github.com/R0rschach02/sleep-motion-tracker.git
cd sleep-motion-tracker
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```
opencv-python
numpy
pandas
```

---

## 🚀 Usage

Run the motion tracker:
```bash
python sleep_motion_tracker.py
```

After running, a `motion_log.csv` file will be generated, containing time-stamped motion data:
```
timestamp,motion_detected,motion_area
2025-10-26 23:45:02,True,5421
2025-10-26 23:45:03,False,0
2025-10-26 23:45:04,True,3240
```

You can visualize the data using **Matplotlib**, **Grafana**, or any CSV analysis tool.

---

## 🔮 Future Work

Planned improvements include:
- Integration with wearable sensors (IMU, pulse oximeter)
- BLE or MQTT data transfer for morning synchronization
- Real-time motion visualization dashboard
- Edge-optimized ML model for posture or sleep-phase recognition
- Multi-sensor fusion (camera + sound analysis)

---

## 📂 Repository Structure

```
sleep-motion-tracker/
│
├── sleep_motion_tracker.py   # Main Python script
├── motion_log.csv            # Example output file
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

---
