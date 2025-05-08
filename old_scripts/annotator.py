import cv2
import numpy as np
import glob
import os

paths = [
    "RBK_TDT17/1_train-val_1min_aalesund_from_start/img1_e",
    "RBK_TDT17/2_train-val_1min_after_goal/img1_e",
    "RBK_TDT17/3_test_1min_hamkam_from_start/img1_e",
    "RBK_TDT17/4_annotate_1min_bodo_start/img1_e"
]

# ----------------- Configuration -----------------
IMAGE_FOLDER = "RBK_TDT17/1_train-val_1min_aalesund_from_start/img1"     # folder containing your original (pre-edge) images
PERFECT_FIELD = "field.jpg"        # perfect field image (same aspect ratio)
OUTPUT_DIR = "../annotations/"  # where homography matrices will be saved
WINDOW_NAME = "Homography Annotator"
MARGIN = 800

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load source image paths
image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))  # adjust pattern

# Load perfect field image and its corners
perfect = cv2.imread(PERFECT_FIELD, cv2.IMREAD_GRAYSCALE)
h, w = perfect.shape[:2]
src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

# Destination points (initially identity), image-relative coords
dst_pts = src_pts.copy()
selected_idx = -1

# Helper to draw overlay and handles on large canvas
def draw_preview(img, H, dst_pts):
    img_h, img_w = img.shape[:2]
    # Canvas dimensions
    canvas_w = img_w + 2*MARGIN
    canvas_h = img_h + 2*MARGIN
    # Create blank canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    # Paste source image at margin offset
    canvas[MARGIN:MARGIN+img_h, MARGIN:MARGIN+img_w] = img

    # Compute homography in canvas coords: translate by margin
    T = np.array([[1,0,MARGIN],[0,1,MARGIN],[0,0,1]], np.float32)
    Hc = T.dot(H)
    # Warp perfect field onto canvas
    warped = cv2.warpPerspective(cv2.cvtColor(perfect, cv2.COLOR_GRAY2BGR), Hc,
                                 (canvas_w, canvas_h))
    # Apply cyan overlay
    mask = warped[:, :, 0] > 0
    canvas[mask] = (255, 255, 0)

    # Draw handles (always visible at canvas positions)
    for (x, y) in dst_pts:
        cx, cy = int(x + MARGIN), int(y + MARGIN)
        cv2.circle(canvas, (cx, cy), 8, (0, 0, 255), -1)
    return canvas

# Mouse event handler on canvas
def mouse_cb(event, x, y, flags, param):
    global selected_idx, dst_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        # select nearest corner (canvas coords)
        canvas_pts = dst_pts + np.array([MARGIN, MARGIN])
        dists = np.linalg.norm(canvas_pts - np.array([x, y]), axis=1)
        idx = np.argmin(dists)
        if dists[idx] < 20:
            selected_idx = idx
    elif event == cv2.EVENT_MOUSEMOVE and selected_idx >= 0:
        # update corner: convert canvas coords back to image-relative
        dst_pts[selected_idx] = [x - MARGIN, y - MARGIN]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_idx = -1

# Main annotation loop
def run_annotator():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # Set a fixed window size based on first image plus margins
    first_img = cv2.imread(image_paths[0])
    fh, fw = first_img.shape[:2]
    cv2.resizeWindow(WINDOW_NAME, fw + 2*MARGIN, fh + 2*MARGIN)
    # Attach mouse callback
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (480, 270))
        img_h, img_w = img.shape[:2]

        # Reset points each image
        global dst_pts
        dst_pts = src_pts.copy()

        while True:
            # Compute homography from src to dst (image-relative)
            H, _ = cv2.findHomography(src_pts, dst_pts.astype(np.float32))

            # Draw preview on large canvas
            preview = draw_preview(img, H, dst_pts)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('n'):
                # Save H
                base = os.path.splitext(os.path.basename(img_path))[0]
                np.savetxt(os.path.join(OUTPUT_DIR, base + '_H.csv'), H, delimiter=',')
                print(f"Saved {base}_H.csv")
                break
            elif key == ord('r'):
                dst_pts = src_pts.copy()
                print("Reset homography to identity")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
            # Nudge selected corner
            elif selected_idx >= 0 and key in [ord('w'), ord('a'), ord('s'), ord('d')]:
                dx = dy = 0
                if key == ord('w'): dy = -5
                elif key == ord('s'): dy = 5
                elif key == ord('a'): dx = -5
                elif key == ord('d'): dx = 5
                dst_pts[selected_idx] += [dx, dy]

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_annotator()
