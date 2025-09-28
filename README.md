# 🧠 Image Analysis Projects with OpenCV

This repository contains two mini-projects focused on image analysis using Python and OpenCV. Both projects demonstrate different real-world applications of computer vision, including classification, object detection, and evaluation techniques.

---

## 📁 Project Structure

```
.
├── Eye_State_Detection_with_OpenCV/    # Eye state detection from video using LBP + classifier
├── Parking_and_eye_detection/          # Parking space classification using LBP + SVM
└── README.md                           # You are here
```


---

## 👁️ 1. Eye State Detection with OpenCV

Detects whether eyes are **open** or **closed** from a video feed using:

- Haar cascade for eye detection
- LBP (Local Binary Pattern) feature extraction
- Classifier training (SVM or OpenCV LBPH)
- Evaluation against annotated ground truth

🔗 [Explore the project →](./Eye_State_Detection_with_OpenCV)

---

## 🅿️ 2. Parking and Eye Detection

Analyzes images of parking lots to classify each spot as **occupied** or **free**. Also includes eye-state detection code.

Features:

- LBP-based feature extraction
- Multiple LBP configurations and methods tested
- SVM and LBPH classifier comparison
- Custom annotation-based evaluation
- Perspective correction (bird’s eye view)

🔗 [Explore the project →](./Parking_and_eye_detection)

---
