import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

train_1_path = "RBK_TDT17/1_train-val_1min_aalesund_from_start/img1"
train_2_path = "RBK_TDT17/2_train-val_1min_after_goal/img1"
train_3_path = "RBK_TDT17/3_test_1min_hamkam_from_start/img1"
train_4_path = "RBK_TDT17/4_annotate_1min_bodo_start/img1"

field_dimensions = {'x': 105.0, 'y': 68.0}

def load_image_paths(folder_path):
    return glob.glob(os.path.join(folder_path, "*.jpg"))

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image {} not found.".format(image_path))
        return None

    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def detect_and_draw_lines(image, canny_thresh1=50, canny_thresh2=150, hough_rho=1, hough_theta=np.pi/180, hough_thresh=50, min_line_len=100, max_line_gap=10):
    # 1. Convert to gray & edge‐detect
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)

    # 2. Run probabilistic Hough
    lines = cv2.HoughLinesP(edges,
                            rho=hough_rho, theta=hough_theta, threshold=hough_thresh,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # 3. Draw lines in cyan (BGR=(255,255,0)), thickness=2
    out = image.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
    return out



def display_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    image_paths = load_image_paths(train_1_path)
    image = preprocess_image(image_paths[0])
    display_image(image)
    image_detect = detect_and_draw_lines(image)
    display_image(image_detect)

    return

main()