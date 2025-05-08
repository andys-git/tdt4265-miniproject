import os
import sys
import yaml
import random
import torch
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# original train model
def train_model(data_config, class_names,
                load_model_file='yolov8n.pt',
                save_model_file='yolo_model_full_0.pt',
                epochs=100,
                device='cpu'):
    """
    Train from a base model.
    - load_model_file: path to pretrained or base model (e.g. 'yolov8n.pt')
    - save_model_file: output path to save trained weights
    """
    if os.path.exists(save_model_file):
        print(f"Model '{save_model_file}' already exists; exiting.")
        sys.exit(0)

    print(f"Using device: {device}")
    model = YOLO(load_model_file)

    print(f"Starting training for {epochs} epochs on device {device}...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=(1280, 720),
        batch=4,
        device=device,
        workers=2,
        augment=True,
        mosaic=0.8,
        mixup=0.3,
        rect=True,
        verbose=True,
    )

    # Print summary metrics
    if hasattr(results, 'metrics') and results.metrics:
        print("Epoch |   loss   |   box    |   cls    |   dfl    |  mAP@0.5")
        print("--------------------------------------------------------------")
        for i, m in enumerate(results.metrics):
            print(f"{i+1:5d} | {m.get('loss',0):7.4f} | {m.get('box',0):7.4f} |"
                  f" {m.get('cls',0):7.4f} | {m.get('dfl',0):7.4f} | {m.get('mAP@0.5',0):9.4f}")

    model.save(save_model_file)
    print(f"✅ Trained model saved to '{save_model_file}'")

    return results



# Resume / fine-tune model
def train_model_resume(data_config, class_names,
                       load_model_file,
                       save_model_file,
                       extra_epochs=100,
                       device='cpu'):
    if not os.path.exists(load_model_file):
        raise FileNotFoundError(f"Checkpoint not found: {load_model_file}")

    print(f"Using device: {device}")
    model = YOLO(load_model_file)

    print(f"Fine-tuning for {extra_epochs} more epochs on device {device}...")
    results = model.train(
        data=data_config,
        epochs=extra_epochs,
        imgsz=(1536, 1024),
        batch=2,
        device=device,
        workers=2,
        augment=True,
        mosaic=0.8,
        mixup=0.3,
        lr0=0.001,
        lrf=0.01,
        resume=False,
        verbose=True,
    )

    if hasattr(results, 'metrics') and results.metrics:
        print("Epoch |   loss   |   box    |   cls    |   dfl    |  mAP@0.5")
        print("--------------------------------------------------------------")
        for i, m in enumerate(results.metrics):
            print(f"{i+1:5d} | {m.get('loss',0):7.4f} | {m.get('box',0):7.4f} |"
                  f" {m.get('cls',0):7.4f} | {m.get('dfl',0):7.4f} | {m.get('mAP@0.5',0):9.4f}")

    model.save(save_model_file)
    print(f"✅ Fine-tuned model saved to '{save_model_file}'")

    return results


# New augmented training function
def train_model_resume_augmented(data_config, class_names,
                                 load_model_file=None,
                                 save_model_file=None,
                                 epochs=100,
                                 device='cpu'):

    # --- Decide whether to start fresh or resume ---
    if load_model_file and os.path.exists(load_model_file):
        print(f"Resuming training from checkpoint '{load_model_file}'.")
        model = YOLO(load_model_file)
    else:
        if save_model_file:
            print(f"No existing checkpoint found at '{load_model_file}', starting from scratch.")
        else:
            print("No save path provided; starting from scratch.")
        model = YOLO("yolov8n.pt")
        # If you want to override class names on a fresh model:
        model.model.names = class_names

    print(f"Using device: {device}")

    # --- Callback to randomly grayscale input with 10% chance ---
    def to_gray(im, path):
        if random.random() < 0.1:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return im

    model.add_callback('load_item', to_gray)

    print(f"Starting augmented training for {epochs} epochs on device {device}...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=(1536, 1024),
        batch=2,
        device=device,
        workers=2,
        # controlled augmentations
        augment=True,
        mosaic=0.5,
        mixup=0.3,
        hsv_h=0.03,
        hsv_s=1.0,
        hsv_v=0.5,
        translate=0.1,
        scale=0.5,
        rect=True,
        resume=bool(save_model_file and os.path.exists(save_model_file)),  # tell Ultralytics to resume
        verbose=True,
    )

    # --- Print training summary ---
    if hasattr(results, 'metrics') and results.metrics:
        print("Epoch |   loss   |   box    |   cls    |   dfl    |  mAP@0.5")
        print("--------------------------------------------------------------")
        for i, m in enumerate(results.metrics):
            print(f"{i + 1:5d} | {m.get('loss', 0):7.4f} | {m.get('box', 0):7.4f} |"
                  f" {m.get('cls', 0):7.4f} | {m.get('dfl', 0):7.4f} | {m.get('mAP@0.5', 0):9.4f}")

    # --- Save final checkpoint if requested ---
    if save_model_file:
        model.save(save_model_file)
        print(f"✅ Model saved to '{save_model_file}'")

    return results


def plot_predictions(model_file, image_path, device='cpu'):
    model = YOLO(model_file)
    results = model.predict(
        source=image_path,
        conf=0.25,
        device=device,
        verbose=False
    )[0]

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    colors = {i:(random.random(),random.random(),random.random())
              for i in range(len(results.names))}

    for box, cls, conf in zip(results.boxes.xyxy,
                               results.boxes.cls,
                               results.boxes.conf):
        x1,y1,x2,y2 = box
        c = int(cls)
        w,h = x2-x1, y2-y1
        ax.add_patch(plt.Rectangle((x1,y1), w, h,
                                   fill=False, linewidth=2,
                                   edgecolor=colors[c]))
        ax.text(x1, y1-5,
                f"{results.names[c]} {conf:.2f}",
                color='white', backgroundcolor=colors[c], fontsize=12)

    ax.imshow(img)
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("[ERROR] No CUDA-enabled GPU detected. Exiting.")
        sys.exit(1)
    device = '0'

    # Standard training
    # train_model('data.yaml', ['player','referee','ball'], 'yolov8n.pt', 'yolo_model_full_0.pt', epochs=100, device=device)

    # Resume training
    # train_model_resume('data.yaml', ['player','referee','ball'], 'yolo_model_full_0.pt',
    #                    'yolo_model_full_1.pt', extra_epochs=50, device=device)

    # augmented training
    # train_model_resume_augmented('data.yaml', ['player','referee','ball'],
    #                              load_model_file='yolo_model_full_1_cpt4.pt',
    #                              save_model_file='yolo_model_full_1_cpt5.pt',
    #                              epochs=200, device=device)

    point_classes = [
            'middle_central_circle',
            'top_central_circle',
            'bottom_central_circle',
            'pitch_top_middle',
            'pitch_top_left_corner',
            'left_penalty_box_top_corner',
            'left_penalty_box_bottom_corner',
            'left_penalty_arc_top',
            'left_penalty_arc_bottom',
            'right_penalty_arc_top',
            'right_penalty_arc_bottom',
            'pitch_top_right_corner',
            'right_penalty_box_top_corner',
            'right_penalty_box_bottom_corner'
        ]

    train_model_resume_augmented('point_data/data.yaml', point_classes,
                                 load_model_file='yolo_model_points_1.pt',
                                 save_model_file='yolo_model_points_2.pt',
                                 epochs=100, device=device)