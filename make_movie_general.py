import cv2
import os
import glob

def images_to_video(input_dir, output_path, fps=24, image_extensions=['*.jpg', '*.png']):
    # Collect image paths
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    # Sort image files by filename
    image_files.sort()

    if not image_files:
        print("No image files found in directory.")
        return

    # Read the first image to get dimensions
    frame = cv2.imread(image_files[0])
    if frame is None:
        print(f"Could not read the first image: {image_files[0]}")
        return

    height, width, _ = frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'avc1' or 'H264' if needed
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        resized_frame = cv2.resize(frame, (width, height))
        video_writer.write(resized_frame)

    video_writer.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    input_directory = r'point_data/images'
    output_video_path = r'collection_check.mp4'
    images_to_video(input_directory, output_video_path, fps=24)
