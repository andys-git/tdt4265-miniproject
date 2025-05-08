import os
import yaml
import glob
import random
import shutil
from pathlib import Path

# === Configuration ===
# Flat source directory containing both images and their .txt labels
SRC_DIR = Path('point_data_clean')
# Base directory under which images/ and labels/ subfolders will be created
OUT_ROOT = Path('point_data')

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# Class metadata
CLASS_NAMES = [
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


def get_pairs(src_dir):
    """Return list of (img_path, lbl_path) for non-empty label siblings."""
    pairs = []
    for ext in EXTS:
        for img_path in src_dir.glob(f'*{ext}'):
            lbl_path = img_path.with_suffix('.txt')
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                pairs.append((img_path, lbl_path))
            else:
                print(f"[WARN] skipping unlabeled or empty: {img_path.name}")
    return sorted(pairs)


def make_dirs(root):
    # Create images/train, images/val, images/test and labels counterparts
    for split in ['train', 'val', 'test']:
        (root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (root / 'labels' / split).mkdir(parents=True, exist_ok=True)


def split_and_copy(pairs, root):
    random.seed(SEED)
    random.shuffle(pairs)
    total = len(pairs)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    splits = [
        ('train', pairs[:n_train]),
        ('val', pairs[n_train:n_train + n_val]),
        ('test', pairs[n_train + n_val:])
    ]
    counts = {}
    for split, items in splits:
        for img_path, lbl_path in items:
            # Copy image
            tgt_img = root / 'images' / split / img_path.name
            shutil.copy(img_path, tgt_img)
            # Copy label
            tgt_lbl = root / 'labels' / split / lbl_path.name
            shutil.copy(lbl_path, tgt_lbl)
        counts[split] = len(items)
    print(f"Distributed {total} pairs → train={counts['train']}, val={counts['val']}, test={counts['test']}")


def write_data_yaml(root):
    data = {
        'train': str(root / 'images' / 'train'),
        'val':   str(root / 'images' / 'val'),
        'test':  str(root / 'images' / 'test'),
        'nc':    len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    yaml_path = root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    print(f"Wrote {yaml_path}")


def main():
    pairs = get_pairs(SRC_DIR)
    print(f"Found {len(pairs)} image+label pairs in {SRC_DIR}")
    make_dirs(OUT_ROOT)
    split_and_copy(pairs, OUT_ROOT)
    write_data_yaml(OUT_ROOT)


if __name__ == '__main__':
    main()