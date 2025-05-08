import cv2
import os
import glob
import json

# ------------- Configuration -------------
IMAGE_FOLDER = './data'    # root folder containing subfolders img1, img1_e, etc.
OUTPUT_FOLDER = './labels' # where to save YOLO .txt files
IMAGE_EXT = '.png'

# Define your classes in order:
CLASSES = [
    'corner_tl', 'corner_tr', 'corner_br', 'corner_bl',
    'penalty1_tl', 'penalty1_tr', 'penalty1_br', 'penalty1_bl',
    'penalty2_tl', 'penalty2_tr', 'penalty2_br', 'penalty2_bl',
    'center_circle'
]

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Mouse drawing state
drawing = False
ix, iy = -1, -1
rects = []   # list of (class_id, x1,y1,x2,y2)
current_image = None
h, w = 0, 0

# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, rects, current_image
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img = current_image.copy()
        cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), 2)
        for cid, x1,y1,x2,y2 in rects:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
        cv2.imshow('label', img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1, x2, y2 = ix, iy, x, y
        # prompt for class
        print('\nDrawn box:', x1,y1,x2,y2)
        for i,name in enumerate(CLASSES): print(f"{i}: {name}")
        cid = int(input('Enter class index: '))
        rects.append((cid, x1, y1, x2, y2))
        # redraw
        img = current_image.copy()
        for cid, x1,y1,x2,y2 in rects:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
        cv2.imshow('label', img)


def save_labels(image_path):
    # Write YOLO-format .txt: class x_center y_center w h (normalized)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(OUTPUT_FOLDER, basename + '.txt')
    lines = []
    for cid, x1,y1,x2,y2 in rects:
        x_c = (x1 + x2) / 2 / w
        y_c = (y1 + y2) / 2 / h
        bw  = abs(x2 - x1) / w
        bh  = abs(y2 - y1) / h
        lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved labels to {label_path}\n")


def main():
    global current_image, rects, h, w
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, f"*/*{IMAGE_EXT}")))
    cv2.namedWindow('label', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('label', mouse_callback)

    for img_path in image_paths:
        # load image
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        current_image = img.copy()
        rects = []
        cv2.imshow('label', img)
        print(f"Annotating: {img_path}")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):
                save_labels(img_path)
                break
            elif key == ord('r'):
                rects = []
                cv2.imshow('label', current_image)
                print("Cleared all boxes.")
            elif key == ord('q'):
                print("Exiting.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Save class names for YOLO config
    with open(os.path.join(OUTPUT_FOLDER, 'classes.json'), 'w') as cf:
        json.dump(CLASSES, cf, indent=2)
    print(f"Saved class list to {os.path.join(OUTPUT_FOLDER, 'classes.json')}")
    main()
