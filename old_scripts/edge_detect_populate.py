import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

paths = [
    "RBK_TDT17/1_train-val_1min_aalesund_from_start/img1",
    "RBK_TDT17/2_train-val_1min_after_goal/img1",
    "RBK_TDT17/3_test_1min_hamkam_from_start/img1",
    "RBK_TDT17/4_annotate_1min_bodo_start/img1"
]

def display_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def load_image_paths(folder_path):
    return glob.glob(os.path.join(folder_path, "*.jpg"))

def detect_edges(image):
    h_img, w_img = image.shape[:2]

    # mask green + boost white + edges
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, (25,40,60), (95,255,255))
    masked = cv2.bitwise_and(image, image, mask=hsv_mask)

    lab2 = cv2.cvtColor(masked, cv2.COLOR_BGR2Lab)
    l_channel = lab2[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l_channel)

    _, white_mask = cv2.threshold(l_eq, 150, 255, cv2.THRESH_BINARY)
    edges = cv2.dilate(
        white_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),
        iterations=3
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    min_area = 800
    edges_clean = np.zeros_like(white_mask)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            edges_clean[labels == lab] = 255

    edges_clean = cv2.resize(edges_clean, (w_img // 4, h_img // 4), interpolation=cv2.INTER_AREA)
    _, edges_clean = cv2.threshold(edges_clean, 127, 255, cv2.THRESH_BINARY)

    return edges_clean


def process_images(image_folders):
    for folder in image_folders:
        output_folder = os.path.join(os.path.dirname(folder), os.path.basename(folder) + "_e")
        os.makedirs(output_folder, exist_ok=True)
        image_paths = load_image_paths(folder)

        for img_path in image_paths:
            print(f"Processing: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load {img_path}")
                continue

            edges = detect_edges(img)
            # display_image(edges)

            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}_e{ext}")
            cv2.imwrite(output_path, edges)

if __name__ == "__main__":
    process_images(paths)
