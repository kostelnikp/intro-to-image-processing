import cv2 as cv    
import numpy as np  
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC

VIDEO_FILE = "eyes/fusek_face_car_01.avi"
EYE_STATE_FILE = "eyes/eye_state.txt"
EYE_CASCADE_PATH = "eyes/eye_cascade_fusek.xml"

eye_cascade = cv.CascadeClassifier(EYE_CASCADE_PATH)

if eye_cascade.empty():
    raise IOError(f"Nepodarilo sa načítať kaskádu očí zo súboru {EYE_CASCADE_PATH}")

def extract_lbp(image, radius, n_points, method):
    lbp = local_binary_pattern(image, n_points, radius, method=method)
    n_bins = int(n_points + 2)
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_features(images, radius, n_points, method):
    features = []
    for image in images:
        extraction = extract_lbp(image, radius, n_points, method)
        features.append(extraction)
    return np.array(features)

def detect_eyes(face_gray, face_color, best_clf):
    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=45,
        minSize=(30, 30)
    )
    for (ex, ey, ew, eh) in eyes:
        eye_region = face_gray[ey:ey+eh, ex:ex+ew]
        
        state = best_clf.predict(eye_region)
        
        if state[0] == 1:
            cv.putText(face_color, "open", (ex, ey-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            color = (0, 255, 0)
        else:
            cv.putText(face_color, "close", (ex, ey-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            color = (0, 0, 255)
        
        cv.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), color, 2)
        cv.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 255), 1)
        
    return state[0]
    

def main():
    video_cap = cv.VideoCapture(VIDEO_FILE)
    if not video_cap.isOpened():
        raise IOError(f"Nepodarilo sa otvoriť video súbor {VIDEO_FILE}")

    frame_count = 0
    images = []
    labels = []

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_color = frame.copy()
        eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=35, minSize=(30, 30))

        with open(EYE_STATE_FILE, 'r') as f:
            eye_states = f.readlines()

        for (ex, ey, ew, eh) in eyes[:1]:  
            eye_roi = gray_frame[ey:ey + eh, ex:ex + ew]
            if frame_count < len(eye_states):
                eye_state = eye_states[frame_count].strip()
                labels.append(1 if eye_state == "open" else 0)
                images.append(eye_roi) 
            cv.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            cv.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 255, 255), 1)
            cv.putText(face_color, "eye", (ex, ey-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            break
        
        frame_count += 1
        
        cv.imshow("Face Detection", face_color)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    print(f"Processed {frame_count} frames and collected {len(images)} labeled eye regions.")
    
    cv.destroyAllWindows()


    X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    X_train_imgs = [cv.resize(img, (50, 50)) for img in X_train_imgs]
    X_test_imgs = [cv.resize(img, (50, 50)) for img in X_test_imgs]
    
    
    lbp_configs = [
        (1, 8),
        (2, 16),
        (3, 24),
        (4, 32),
        (5, 40),
        (6, 48),
        (7, 56),
        (8, 64)
    ]
    
    lbp_methods = ["default", "ror", "uniform", "opencv"]
    
    best_config = None
    best_method = None
    accuracy = 0
    best_accuracy = 0
    best_clf = None
    
    for method in lbp_methods:
        print(f"Používanie metódy LBP: {method}")
        for radius, n_points in lbp_configs:
            start_time = time.time()
            if method == "opencv" and (radius, n_points) not in lbp_configs[:1]:
                continue
            print(f"Spracovanie LBP s parametrami: radius={radius}, n_points={n_points}")

            if method == "opencv":
                opencv_recognizer = cv.face.LBPHFaceRecognizer_create(radius=radius, neighbors=n_points, grid_x=8, grid_y=8)
                opencv_recognizer.train(np.array(X_train_imgs), np.array(y_train))
                predictions = []
                for img in X_test_imgs:
                    pred, conf = opencv_recognizer.predict(img)
                    predictions.append(pred)
                accuracy = accuracy_score(y_test, predictions)
                if accuracy > best_accuracy:
                    best_clf = opencv_recognizer
            else:
                X_train_features = extract_features(X_train_imgs, radius, n_points, method)
                X_test_features = extract_features(X_test_imgs, radius, n_points, method)
                clf = SVC(kernel="linear", random_state=42)
                clf.fit(X_train_features, y_train)
                y_pred = clf.predict(X_test_features)
                accuracy = accuracy_score(y_test, y_pred)
                if accuracy > best_accuracy:
                    best_clf = clf
            
            process_time = time.time() - start_time
            print(f"Presnosť: {accuracy*100:.2f}%")
            print(f"Čas spracovania: {process_time:.2f} sekúnd\n")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = (radius, n_points)
                best_method = method
    
    print(f"Najlepšia konfigurácia: radius={best_config[0]}, n_points={best_config[1]}, method={best_method}")
    print(f"Najlepšia presnosť: {best_accuracy*100:.2f}%")
    
    
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    txt_file = open(EYE_STATE_FILE, "r")
    
    video_cap = cv.VideoCapture(VIDEO_FILE)
    if not video_cap.isOpened():
        print("Nepodarilo sa otvoriť video súbor.")
        return
    
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_copy = frame.copy()
        eye_states = []
        
        start_time = time.time()
        
        eye_state = detect_eyes(gray_frame, frame_copy, best_clf)
        
        if eye_state == 1:
            if txt_file.readline().strip() == "open":
                total_tp += 1
            else:
                total_fp += 1
        else:
            if txt_file.readline().strip() == "close":
                total_tn += 1
            else:
                total_fn += 1
        
        
        end_time = time.time()
        detection_time = (end_time - start_time) * 1000
        cv.putText(frame_copy, f"Trvanie detekcie oci: {detection_time:.2f}ms", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow("Face Detection", frame_copy)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    total_accuracy = (total_tp + total_tn) / (total_tp +
                                              total_tn + total_fp + total_fn) * 100
    print(f"Celková presnosť: {total_accuracy:.2f}%")

    

if __name__ == "__main__":
    main()