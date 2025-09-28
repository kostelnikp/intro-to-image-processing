import sys
import cv2 as cv
import os
import numpy as np
from collections import deque
import math

IMAGE_FOLDER = "anomal_hd_30fps_01"
FACE_CASCADE_PATH = "lbpcascade_frontalface_improved.xml"
FACE_MARK_MODEL_PATH = "opencv_LBF686_GTX.yaml"
ANOTATION_FILE = "anomal_hd_30fps_01/anot.txt"

face_cascade = cv.CascadeClassifier(FACE_CASCADE_PATH)
face_mark = cv.face.createFacemarkLBF()
face_mark.loadModel(FACE_MARK_MODEL_PATH)

THRESHOLD = 0.26
USED_FRAMES = 5
EAR_HISTORY = deque(maxlen=USED_FRAMES)  

CONSECUTIVE_FRAMES = math.ceil(0.5 * USED_FRAMES)
closed_eyes_counter = 0

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_EAR(eye):
    ear = (euclidean_distance(eye[1], eye[5]) + euclidean_distance(eye[2], eye[4])) / (2.0 * euclidean_distance(eye[0], eye[3]))
    return ear

def detect_facial_landmarks(image):
    global closed_eyes_counter
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.2,
        minNeighbors=3,
        minSize=(50, 50)
    )

    if len(faces) > 0:
        status, landmarks = face_mark.fit(image, faces)
        for f in range(len(landmarks)):
            cv.face.drawFacemarks(image, landmarks[f], color=(255, 255, 255))
            
            pts = landmarks[f]
            if pts.ndim == 3:
                pts = pts[0]
            
            left_eye = pts[36:42]  
            right_eye = pts[42:48] 
                        
            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            
            ear = (left_ear + right_ear) / 2.0
            
            if ear < THRESHOLD:
                closed_eyes_counter += 1
                EAR_HISTORY.append(0)
            else:
                closed_eyes_counter = 0
                EAR_HISTORY.append(1) 
            
            if closed_eyes_counter >= CONSECUTIVE_FRAMES:
                cv.putText(
                    image,
                    "Eyes Closed",
                    (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv.LINE_AA
                )
            
            (x, y, w, h) = faces[f]
            cv.putText(
                image,
                f"EAR: {ear:.2f}",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA
            )
    
    history_text = f"{list(EAR_HISTORY)}"
    cv.putText(
        image,
        history_text,
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )
    
    for one_face in faces:
        cv.rectangle(image, one_face, (0, 0, 255), 12)
        cv.rectangle(image, one_face, (255, 255, 255), 2)

    cv.imshow("image", image)

def load_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                start, end = map(int, parts)
                annotations.append((start, end)) 
    return annotations

def verify_prediction(frame_number, prediction, annotations):
    for start, end in annotations:
        if start <= frame_number <= end:
            return prediction == 0 
    return prediction == 1 

def main():
    annotations = load_annotations(ANOTATION_FILE)
    correct_predictions = 0
    total_predictions = 0

    for frame_number, image_file in enumerate(sorted(os.listdir(IMAGE_FOLDER))):
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        if image is None:
            continue

        detect_facial_landmarks(image)

        prediction = 1 if sum(EAR_HISTORY) > 0 else 0
        result = verify_prediction(frame_number, prediction, annotations)

        if result is not None:
            total_predictions += 1
            if result:
                correct_predictions += 1

        cv.waitKey(2)

    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No matching annotations found.")

if __name__ == "__main__":
    main()
    sys.exit(0)
