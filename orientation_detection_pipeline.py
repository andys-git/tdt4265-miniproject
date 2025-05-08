import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from generate_field import get_field_markers
import matplotlib.pyplot as plt
import pickle as pkl
from make_movie_general import images_to_video
from break_colinearity import *

# Configuration
FIELD_IMG = "field.jpg"
YOLO_MODEL_POINTS = "yolo_model_points_2.pt"
FIELD_DIMENSIONS = {'x': 105.0, 'y': 68.0}
FIELD_MARKERS = get_field_markers(FIELD_DIMENSIONS)

# Detection thresholds
CONF_THRESH = {
    "player": 0.25,
    "referee": 0.25,
    "ball": 0.20
}

# Homography parameters
MIN_MARKERS = 4
RANSAC_THRESH = 3.0
MAX_INTERP_GAP = 10
SMOOTH_WINDOW = 11
REPROJ_THRESH = 0.5


def natural_sort_key(fname):
    nums = re.findall(r'\d+', fname)
    return int(nums[0]) if nums else float('inf')


def load_frames(folder):
    imgs = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    return sorted(imgs, key=natural_sort_key)


def detect_field_points(model, frames, folder):
    # print("Running model...")
    frame_dict = {}
    for i, f in enumerate(frames):
        print(f"(keypoint) Processing frame {i + 1} out of {len(frames)} total")
        res = model.predict(source=os.path.join(folder, f), conf=0.0, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        pts = []
        for (x1, y1, x2, y2), cls, cf in zip(boxes, classes, confs):
            label = res.names[cls]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            pts.append((label, float(cf), cx, cy))
        frame_dict[f] = {'points': pts, 'warp': None}
    return frame_dict

def estimate_homographies(frame_dict, frames, field_markers, image_path,
                          min_markers=4, min_inliers=4,
                          allow_180=True, min_confidence = 0.6):
    """
    Compute a homography per frame (image → world) if enough markers are found,
    reject any that flips the field axes, optionally fix 180° flips,
    and linearly interpolate any missing Hs.

    Returns Hs: a list where Hs[i] is the 3×3 homography for frames[i].
    """

    def is_upright(H):
        A = (H / H[2, 2])[:2, :2]
        return np.linalg.det(A) > 0

    def fix_or_none(H):
        Hn = H / H[2,2]
        if is_upright(Hn):
            return Hn
        if not allow_180:
            return None
        H2 = (R180 @ Hn)
        H2 /= H2[2,2]
        return H2 if is_upright(H2) else None

    def decompose_affine(H):
        """
        Approximate H by its top-2×3 affine part, then
        SVD-decompose the 2×2 linear part into R·S.
        Returns (angle, scale_x, scale_y, trans_x, trans_y).
        """
        A = H[:2, :2]
        U, S, Vt = np.linalg.svd(A)
        R = U @ Vt
        angle = np.arctan2(R[1, 0], R[0, 0])  # rotation
        scale_x, scale_y = S  # singular values
        tx, ty = H[0, 2], H[1, 2]  # translation
        return np.array([angle, scale_x, scale_y, tx, ty])

    def smooth_homographies(Hs, window_size=5):
        """
        Simple moving‐average smoothing of a sequence of homographies.
        window_size: number of frames to each side (total window = 2*window_size+1)
        """
        import numpy as np

        N = len(Hs)
        Hs_sm = [None] * N

        for i in range(N):
            # determine window bounds
            lo = max(0, i - window_size)
            hi = min(N - 1, i + window_size)

            # accumulate
            H_acc = np.zeros_like(Hs[0])
            count = 0
            for j in range(lo, hi + 1):
                H_acc += Hs[j]
                count += 1

            H_avg = H_acc / float(count)
            # renormalize homogeneous scale
            H_avg = H_avg / H_avg[2, 2]
            Hs_sm[i] = H_avg

        return Hs_sm


    for fname in frames:
        frame_dict[fname]['points'] = [
            p for p in frame_dict[fname]['points']
            if p[1] >= min_confidence
        ]

    target_labels = {'middle_central_circle', 'top_central_circle', 'bottom_central_circle', 'pitch_top_middle'}

    for fname in frames:
        pts = frame_dict[fname]['points']
        if len(pts) == 4 and {p[0] for p in pts} == target_labels:
            new_pts = inject_synthetic_points(fname, frame_dict, image_path)
            frame_dict[fname]['points'].extend(new_pts)
            continue

    # --- Precompute world corners & 180° flip in world-space ---
    wpts = np.vstack(list(field_markers.values()))
    xs, ys = wpts[:,0], wpts[:,1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    corners_world = np.array([
        [min_x, min_y], [max_x, min_y],
        [max_x, max_y], [min_x, max_y]
    ], np.float32).reshape(-1,1,2)
    cx, cy = (min_x+max_x)/2, (min_y+max_y)/2
    T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], np.float32)
    R  = np.array([[-1,0,0],[0,-1,0],[0,0,1]], np.float32)
    T2 = np.array([[1,0, cx],[0,1, cy],[0,0,1]], np.float32)
    R180 = T2 @ R @ T1

    # --- Estimate per-frame homographies ---
    Hs = []
    for fname in frames:
        pts = frame_dict[fname]['points']
        if len(pts) < min_markers:
            Hs.append(None); continue

        img_pts, world_pts = [], []
        for label, _, x, y in pts:
            for wx, wy in field_markers.get(label, []):
                img_pts.append([x,y]); world_pts.append([wx,wy])
        if len(img_pts) < min_markers:
            Hs.append(None); continue

        img_np   = np.array(img_pts,   np.float32)
        world_np = np.array(world_pts, np.float32)
        H, mask = cv2.findHomography(
            img_np,
            world_np,
            cv2.RANSAC,
            10.0,
            maxIters=5000,
            confidence=0.99
        )
        if H is None or int(mask.sum()) < min_inliers:
            print("RANSAC failed")
            Hs.append(None)
        else:
            Hs.append(fix_or_none(H))

    # gather all parameter-vectors
    params = np.array([decompose_affine(H)
                       for H in Hs if H is not None])

    # median and median absolute deviation
    med = np.median(params, axis=0)
    mad = np.median(np.abs(params - med), axis=0)

    # set threshold at, say, 3×MAD on each parameter
    thresh = 3.0 * mad

    # Detect and reject outliers
    for i, H in enumerate(Hs):
        if H is None:
            continue

        # 1) Reflection / flip test
        A = H[:2, :2]
        if np.linalg.det(A) < 0:
            print(f"frame {i}: reflection detected → drop")
            Hs[i] = None
            continue

        # 2) “3×MAD” test as before
        p = decompose_affine(H)
        if np.any(np.abs(p - med) > thresh):
            print(f"frame {i}: global outlier → drop")
            Hs[i] = None

    # Reject any remaining unreasonable jumps frame to frame
    # first, collect all consecutive diffs on your existing good frames
    diffs = []
    prev = None
    for H in Hs:
        if H is not None:
            if prev is not None:
                diffs.append(np.linalg.norm(H - prev, ord='fro'))
            prev = H

    # compute robust threshold = median + 3×MAD
    med = np.median(diffs)
    mad = np.median(np.abs(diffs - med))
    thresh = med + 3 * mad

    # now reject any frame whose jump is too big
    last_valid = None
    for i, H in enumerate(Hs):
        if H is None:
            continue
        if last_valid is not None:
            jump = np.linalg.norm(H - Hs[last_valid], ord='fro')
            if jump > thresh:
                print(f"frame {i}: Frobenius jump {jump:.1f} > {thresh:.1f} → drop")
                Hs[i] = None
                continue
        last_valid = i

    # Interpolate any None entries
    n = len(Hs)
    i = 0

    while i < n:
        if Hs[i] is None:
            # find next valid forward
            j = i + 1
            while j < n and Hs[j] is None:
                j += 1

            # find the last valid backward
            prev = i - 1
            while prev >= 0 and Hs[prev] is None:
                prev -= 1

            if j < n:
                # forward-valid exists at j
                H_next = Hs[j]
                # pick H_prev = last valid if any, else fallback to H_next
                H_prev = Hs[prev] if prev >= 0 else H_next
                gap_len = j - i + 1  # include H_next in the normalization

                # linearly interpolate Hs[i]..Hs[j-1]
                for k in range(i, j):
                    t = (k - prev) / gap_len
                    Hk = (1 - t) * H_prev + t * H_next
                    Hs[k] = Hk / Hk[2, 2]

                i = j  # continue after the gap
            else:
                # no forward-valid (we ran off the end): use last valid if found
                if prev >= 0:
                    H_last = Hs[prev]
                    for k in range(i, n):
                        Hs[k] = H_last.copy()
                # (if prev < 0, there's literally no valid H at all; we leave them as None)
                break
        else:
            i += 1

    Hs = smooth_homographies(Hs, window_size=10)
    return Hs

def visualize_and_save(frame_name, info, input_folder, field_img, field_dim, save_dir):
    print(f"Visualizing frame {frame_name}...")

    # 1) grab and print the homography
    H = info['warp'] / info['warp'][2,2]   # normalize for readability

    # 2) load images
    orig = cv2.imread(os.path.join(input_folder, frame_name))
    mask = cv2.imread(field_img, cv2.IMREAD_GRAYSCALE)
    h_img, w_img = orig.shape[:2]
    h_f, w_f     = mask.shape

    # 3) compute scale from field-dims → mask-pixels
    S = np.array([
        [w_f/field_dim['x'], 0,                0],
        [0,                  h_f/field_dim['y'], 0],
        [0,                  0,                  1]
    ], np.float32)

    # warp mask down into image to overlay
    H_f2i     = np.linalg.inv(S @ H)
    warp_mask = cv2.warpPerspective(mask, H_f2i, (w_img, h_img),
                                    flags=cv2.INTER_NEAREST)

    over = orig.copy()
    over[warp_mask > 0] = (0,255,0)

    # warp the camera view top-down
    top = cv2.warpPerspective(orig, S @ H, (w_f, h_f),
                              flags=cv2.INTER_LINEAR)

    # 4) plot
    fig = plt.figure(figsize=(14, 6), dpi=100)
    for idx, img, title in [(1, over, 'Overlay'), (2, top, 'Top-Down')]:
        ax = fig.add_subplot(1, 2, idx)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    # 5) add the H‐matrix as text below the plots
    mat_txt = '\n'.join(
        [' '.join([f"{val: .3f}" for val in row]) for row in H]
    )
    fig.text(
        0.5, 0.01,
        "H =\n" + mat_txt,
        ha='center', va='bottom',
        fontsize=9, family='monospace'
    )

    # 6) save
    out_path = os.path.join(save_dir, os.path.splitext(frame_name)[0] + '.jpg')
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # leave room at bottom for matrix
    plt.savefig(out_path)
    plt.close(fig)


def save_frame_dict(d, fname='frame_dict.pkl'):
    with open(fname, 'wb') as f: pkl.dump(d, f)

def load_frame_dict(fname='frame_dict.pkl'):
    with open(fname, 'rb') as f: return pkl.load(f)


def main():
    INPUT_FOLDER = r"C:\Users\andys\Documents\TDT4265\pipeline_test_images"
    SAVE_VISUALIZATION_FOLDER = r"C:\Users\andys\Documents\TDT4265\showcase\hamkam_overlay_result"

    model = YOLO(YOLO_MODEL_POINTS)
    frames = load_frames(INPUT_FOLDER)
    try:
        frame_dict = load_frame_dict()
    except FileNotFoundError:
        frame_dict = detect_field_points(model, frames, INPUT_FOLDER)
        save_frame_dict(frame_dict)

    Hs = estimate_homographies(frame_dict, frames, FIELD_MARKERS, INPUT_FOLDER)

    # print(frame_dict)
    for f, H in zip(frames, Hs): frame_dict[f]['warp'] = H

    # for a in Hs:
    #     print(a)

    os.makedirs(SAVE_VISUALIZATION_FOLDER, exist_ok=True)
    for fname, info in frame_dict.items():
        if info['warp'] is not None:
            visualize_and_save(fname, info, INPUT_FOLDER, FIELD_IMG, FIELD_DIMENSIONS, SAVE_VISUALIZATION_FOLDER)

    images_to_video(SAVE_VISUALIZATION_FOLDER, 'overlay_test_set_colinear.mp4', fps=24)

if __name__ == '__main__':
    main()
