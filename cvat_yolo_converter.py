import xml.etree.ElementTree as ET
from pathlib import Path
import os

# COCO_XML    = 'clean_data/hamkam_annotations.xml'
# IMG_DIR     = 'clean_data/hamkam'
# OUT_LABELS  = 'clean_data/hamkam'

# COCO_XML    = 'clean_data/aalesund_annotations.xml'
# IMG_DIR     = 'clean_data/aalesund'
# OUT_LABELS  = 'clean_data/aalesund'

# COCO_XML    = 'clean_data/aftergoal_annotations.xml'
# IMG_DIR     = 'clean_data/aftergoal'
# OUT_LABELS  = 'clean_data/aftergoal'

COCO_XML    = 'point_data_meta/annotations.xml'
IMG_DIR     = 'point_data_clean'
OUT_LABELS  = 'point_data_clean'

EPSILON     = 0.05
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

tree = ET.parse(COCO_XML)
root = tree.getroot()
os.makedirs(OUT_LABELS, exist_ok=True)

for img in root.findall('image'):
    fname = img.attrib['name']
    w = float(img.attrib['width'])
    h = float(img.attrib['height'])
    img_path = Path(IMG_DIR)/fname
    if not img_path.exists():
        print(f"[ERROR] Missing {img_path}, skipping")
        continue

    lines = []
    # Boxes (rectangles)
    for box in img.findall("box"):
        cls = CLASS_NAMES.index(box.attrib['label'])
        xtl = float(box.attrib['xtl'])
        ytl = float(box.attrib['ytl'])
        xbr = float(box.attrib['xbr'])
        ybr = float(box.attrib['ybr'])
        bw, bh = xbr-xtl, ybr-ytl
        x_c = (xtl + bw/2)/w
        y_c = (ytl + bh/2)/h
        lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {bw/w:.6f} {bh/h:.6f}")

    # Points
    for pt in img.findall("points"):
        cls = CLASS_NAMES.index(pt.attrib['label'])
        first_pt = pt.attrib['points'].split(';')[0]
        x_str, y_str = first_pt.split(',')
        x, y = float(x_str), float(y_str)
        x_c, y_c = x / w, y / h
        lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {EPSILON:.6f} {EPSILON:.6f}")

    out = Path(OUT_LABELS)/(Path(fname).stem + '.txt')
    with open(out, 'w') as f:
        f.write("\n".join(lines))
print("Converted CVAT-XML → YOLO labels in", OUT_LABELS)
