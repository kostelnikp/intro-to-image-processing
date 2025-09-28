import os
import time
import numpy as np
import cv2 as cv
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import glob

DATASET_PATH = "parking/"

def load_dataset(dataset_path):
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        label_folder = os.path.join(dataset_path, label)
        if os.path.isdir(label_folder):
            if label.lower() == "free":
                label_val = 0
            elif label.lower() == "full":
                label_val = 1
            else:
                continue
            
            for file in os.listdir(label_folder):
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(label_folder, file)
                    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                    if image is not None:
                        images.append(image)
                        labels.append(label_val)
                        
    return images, np.array(labels)

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

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, one_c):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def main():
    images, labels = load_dataset(DATASET_PATH)
    
    X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42)
    
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
    
    pkm_file = open('parking/parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("parking/test_images_zao/*.jpg")]
    test_images.sort()
    
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    

    for img_name in test_images:
        img_orig = cv.imread(img_name)
        img_copy = img_orig.copy()
        txt_name = img_name.split(".")[0] + ".txt"
        txt_file = open(txt_name, "r")
        start_time = time.time()
        tp = 0
        tn = 0
        fp = 0
        fn = 0
       
        for coor in pkm_coordinates:

            p1 = (int(coor[0]), int(coor[1]))
            p2 = (int(coor[2]), int(coor[3]))
            p3 = (int(coor[4]), int(coor[5]))
            p4 = (int(coor[6]), int(coor[7]))

            warped = four_point_transform(img_copy, coor)
            
            warped = cv.resize(warped, (85, 85))
            warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
            
            status = best_clf.predict(warped)
            
            predicted_status = status[0]

            if predicted_status == 1:
                if txt_file.readline().strip() == "1":
                    tp += 1
                    total_tp += 1
                else:
                    fp += 1
                    total_fp += 1
                color = (0, 0, 255)
            else:
                if txt_file.readline().strip() == "0":
                    tn += 1
                    total_tn += 1
                else:
                    fn += 1
                    total_fn += 1
                color = (0, 255, 0)
               

            cv.line(img_copy, p1, p2, color, 2)
            cv.line(img_copy, p2, p3, color, 2)
            cv.line(img_copy, p3, p4, color, 2)
            cv.line(img_copy, p4, p1, color, 2)
            
        end_time = time.time()
        process_time = (end_time - start_time)
        
        cv.putText(img_copy, f"Cas spracovania: {process_time:.2f}s", (30, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        cv.putText(img_copy, f"Presnost: {accuracy:.2f}%", (30, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv.imshow("Parking map", img_copy)
        cv.waitKey()
    
    total_accuracy = (total_tp + total_tn) / (total_tp +
                                              total_tn + total_fp + total_fn) * 100
    print(f"Celková úspešnosť: {total_accuracy:.2f}")


    
    
if __name__ == "__main__":
    main()
