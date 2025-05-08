import cv2
from pathlib import Path

paths = [
    "RBK_TDT17/1_train-val_1min_aalesund_from_start/img1_e",
    "RBK_TDT17/2_train-val_1min_after_goal/img1_e",
    "RBK_TDT17/3_test_1min_hamkam_from_start/img1_e",
    "RBK_TDT17/4_annotate_1min_bodo_start/img1_e"
]

def create_video_from_images(input_folder, output_file, fps=30):
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")

    # Get all .jpg files sorted by name
    images = sorted([img for img in input_path.glob("*.jpg")])
    if not images:
        raise ValueError("No .jpg images found in the specified folder.")

    first_frame = cv2.imread(str(images[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 format
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    print(f"Video saved to {output_file}")

def create_video_from_images_with_overlay(input_folder, output_file, fps=30):
    input_path = Path(input_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")

    # Get all .jpg files sorted by name
    images = sorted([img for img in input_path.glob("*.jpg")])
    if not images:
        raise ValueError("No .jpg images found in the specified folder.")

    first_frame = cv2.imread(str(images[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 format
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    for folder in paths:
        create_video_from_images(folder, f"{folder.split('/')[1]}_edges.mp4")

