# Updated create_video_from_predictions.py with numeric sorting and debug prints

import os
import re
import random
import cv2
from ultralytics import YOLO

def natural_sort_key(path):
    """Extract numeric parts for natural sorting."""
    fname = os.path.basename(path)
    nums = re.findall(r'\d+', fname)
    return int(nums[0]) if nums else float('inf')

def make_video_yolo(input_folder,
                    model_path,
                    offset,
                    num_images,
                    output_video=None,
                    image_output_dir=None,
                    classes_to_draw=None,
                    fps=30,
                    conf_thresh=0.25):
    # Collect and sort image paths
    img_exts = ['.jpg', '.png']
    imgs = [os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(tuple(img_exts))]
    imgs = sorted(imgs, key=natural_sort_key)[offset:offset + num_images]
    if not imgs:
        raise ValueError(f"No images found in {input_folder}")

    # Load model
    model = YOLO(model_path)

    # Prepare video writer if needed
    writer = None
    if output_video:
        first = cv2.imread(imgs[0])
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # Prepare image output directory if needed
    if image_output_dir:
        os.makedirs(image_output_dir, exist_ok=True)

    # Determine which classes to draw
    classes_set = set(classes_to_draw) if classes_to_draw else None

    # Prepare colors
    colors = {}
    model_names = model.names
    target_classes = classes_set or model_names.values()
    for idx, name in model_names.items():
        if not classes_set or name in classes_set:
            colors[idx] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    # Process each image
    for idx, img_path in enumerate(imgs):
        results = model.predict(source=img_path, conf=conf_thresh, verbose=False)[0]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            cls = int(cls)
            name = model_names[cls]
            if classes_set and name not in classes_set:
                continue
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(cls, (255,255,255))
            label = f"{name} {conf:.2f}"
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert back to BGR for saving
        out_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Write to video if enabled
        if writer:
            writer.write(out_bgr)

        # Save individual frame if requested
        if image_output_dir:
            frame_name = os.path.basename(img_path)
            name_no_ext = os.path.splitext(frame_name)[0]
            save_frame_path = os.path.join(image_output_dir, f"{name_no_ext}.jpg")
            cv2.imwrite(save_frame_path, out_bgr)

    if writer:
        writer.release()
        print(f"Saved video to {output_video}")
    if image_output_dir:
        print(f"Saved frames to {image_output_dir}")

if __name__ == "__main__":
    # === Parameters ===
    input_folder = r"C:\Users\andys\Documents\TDT4265\clean_data\aftergoal"
    model_path   = "yolo_model_full_1_cpt5.pt"
    offset       = 0
    num_images   = 300
    output_video = "showcase/predictions_aftergoal_final_objects.mp4"
    fps          = 25
    conf_thresh  = 0.3

    make_video_yolo(
        input_folder=input_folder,
        model_path=model_path,
        offset=offset,
        num_images=num_images,
        output_video=output_video,
        classes_to_draw=['referee', 'ball'],
        fps=fps,
        conf_thresh=conf_thresh
    )
