import os
import shutil
from PIL import Image

# === Configuration ===
input_folders = [r"C:\Users\andys\Documents\TDT4265\RBK_TDT17\1_train-val_1min_aalesund_from_start\img1",
                 r"C:\Users\andys\Documents\TDT4265\RBK_TDT17\2_train-val_1min_after_goal\img1",
                 r"C:\Users\andys\Documents\TDT4265\RBK_TDT17\3_test_1min_hamkam_from_start\img1"]  # list of input directories
output_base = r"C:\Users\andys\Documents\TDT4265\point_data\images"  # output directory
sequence_length = 50  # number of consecutive images per sequence
step = 100  # number of images to skip between sequences
image_extensions = {".jpg"}  # allowed image extensions

# === Create output base directory ===
os.makedirs(output_base, exist_ok=True)

image_counter = 0

for folder in input_folders:
    all_files = sorted(os.listdir(folder))
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]

    i = 0
    while i + sequence_length <= len(image_files):
        for j in range(sequence_length):
            image_path = os.path.join(folder, image_files[i + j])
            img = Image.open(image_path)
            image_counter += 1
            save_path = os.path.join(output_base, f"img_{image_counter:06d}{os.path.splitext(image_files[i + j])[1]}")
            img.save(save_path)

        i += step

print("Sampling complete.")
