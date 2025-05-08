import yaml, random, os, glob

# === your folders ===
raw_train_dirs = [
    "clean_data/aalesund",
    "clean_data/aftergoal"
]
raw_test_dirs = ["clean_data/hamkam"]

# Optional: max images per folder (None = no subsampling)
MAX_IMAGES_PER_FOLDER = 500
# Split ratio for train/val if needed (applied per-folder in raw_train_dirs)
SPLIT_RATIO = 0.8

EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def get_labeled_images(folder):
    """
    Return sorted list of image paths in `folder` that have a non-empty .txt label sibling.
    """
    imgs = []
    folder = os.path.abspath(folder)
    for ext in EXTS:
        pattern = os.path.join(folder, f'*{ext}')
        for img in glob.glob(pattern):
            lbl = os.path.splitext(img)[0] + '.txt'
            if os.path.isfile(lbl) and os.path.getsize(lbl) > 0:
                imgs.append(img)
            else:
                print(f"[WARN] skipping unlabeled or empty file: {img}")
    return sorted(imgs)


def subsample_list(lst, max_count):
    if max_count and len(lst) > max_count:
        return random.sample(lst, max_count)
    return lst


def main():
    random.seed(42)
    train_imgs = []
    val_imgs = []
    test_imgs = []

    # Process each train folder
    for d in raw_train_dirs:
        imgs = get_labeled_images(d)
        print(f"Found {len(imgs)} labeled images in {d}")  # report count per folder
        imgs = subsample_list(imgs, MAX_IMAGES_PER_FOLDER)
        random.shuffle(imgs)
        split_idx = int(len(imgs) * SPLIT_RATIO)
        train_imgs.extend(imgs[:split_idx])
        val_imgs.extend(imgs[split_idx:])

    # Process test folders
    for d in raw_test_dirs:
        imgs = get_labeled_images(d)
        print(f"Found {len(imgs)} labeled images in {d}")
        imgs = subsample_list(imgs, MAX_IMAGES_PER_FOLDER)
        test_imgs.extend(imgs)

    # Write train.txt, val.txt, test.txt
    for name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        out_file = f'{name}.txt'
        with open(out_file, 'w') as f:
            for path in img_list:
                f.write(path + '\n')
        print(f"Wrote {len(img_list)} entries to {out_file}")

    # Write data.yaml
    data = {
        'train': 'train.txt',
        'val':   'val.txt',
        'test':  'test.txt',
        # total classes: 8 (3 objects + 5 field markers)
        'nc':    8,
        'names': [
            'pitch_corner',
            'penalty_area_corner',
            'halfway_line_intersection',
            'center_circle_point',
            'penalty_spot',
            'player',
            'referee',
            'ball'
        ]
    }
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f)
        print("Wrote data.yaml")
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f)
        print("Wrote data.yaml")

if __name__ == '__main__':
    main()

