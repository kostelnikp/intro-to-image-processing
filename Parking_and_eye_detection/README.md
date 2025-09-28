# Image Analysis Projects with OpenCV

This repository contains two Python-based image analysis systems:

1. **Parking Spot Detection** (`parking.py`)  
   Classifies parking spaces as "free" or "occupied" using LBP (Local Binary Patterns) and SVM classifiers trained on real image patches. Supports test-time perspective correction, evaluation, and performance reporting.

2. **Eye State Detection** (`eye.py`)  
   Detects eye states ("open"/"closed") from video using Haar cascade detection and classifies regions using LBP features. Designed for applications like drowsiness detection or biometric preprocessing.

---

## 🧰 Key Techniques

- Local Binary Pattern (LBP) feature extraction
- SVM and OpenCV LBPH classifiers
- Haar cascade-based region detection (face/eyes)
- Video and image processing with OpenCV
- Custom accuracy evaluation & annotation comparison

---

## 📂 Folders & Files

- 📁 parking/ # Image dataset for parking
- 📁 eyes/ # Eye detection video + annotations
- 📄 parking.py # Parking space classifier
- 📄 eye.py # Eye state classifier

---

## ✅ Dependencies

- Python 3.x
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- scikit-learn
- scikit-image
- numpy