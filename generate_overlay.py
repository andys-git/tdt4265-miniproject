import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def unwarp_all(frames_dict):
    for fname, info in frames_dict.items():
        H = info.get('warp', None)
        if H is None:
            continue

        # helper to project a single (x_img, y_img) → (x_world, y_world)
        def to_world(xy):
            p = np.array([[[xy[0], xy[1]]]], dtype=np.float32)  # shape (1,1,2)
            w = cv2.perspectiveTransform(p, H)[0,0]
            return float(w[0]), float(w[1])

        new_pts = []
        for cls, conf, x, y in info.get('points', []):
            wx, wy = to_world((x, y))
            new_pts.append((cls, wx, wy))
        info['points'] = new_pts

        raw_objs = info.get('objects', {})
        obj_list = raw_objs.get('objects', []) if isinstance(raw_objs, dict) else raw_objs

        new_objs = []
        for cls, conf, x, y in obj_list:
            wx, wy = to_world((x, y))
            new_objs.append((cls, wx, wy))
        info['objects'] = new_objs

    return frames_dict


def plot_frames_on_field(frame_dict,
                         image_folder_path,
                         field_dimensions,
                         background_path='field.jpg',
                         output_dir='output_plots'):
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Load background once
    bg_bgr = cv2.imread(background_path)
    if bg_bgr is None:
        raise FileNotFoundError(f"Background image not found: {background_path}")
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    bh, bw = bg_rgb.shape[:2]

    # Color map for classes
    color_map = {
        'player': 'blue',
        'referee': 'red',
        'ball': 'yellow'
    }

    # Iterate frames
    for fname, info in frame_dict.items():
        # Load original
        orig_path = os.path.join(image_folder_path, fname)
        orig_bgr = cv2.imread(orig_path)
        if orig_bgr is None:
            print(f"Warning: could not load {orig_path}, skipping.")
            continue
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # Prepare scatter data
        xs, ys, cs = [], [], []
        for cls, xw, yw in info.get('objects', []):
            # map world -> pixel
            xp = xw / field_dimensions['x'] * bw
            yp = (field_dimensions['y'] - yw) / field_dimensions['y'] * bh
            xs.append(xp)
            ys.append(yp)
            cs.append(color_map.get(cls, 'white'))

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(8, 12))
        axes[0].imshow(orig_rgb)
        axes[0].axis('off')
        axes[0].set_title(fname)

        axes[1].imshow(bg_rgb)
        axes[1].scatter(xs, ys, c=cs, s=80, edgecolors='black')
        axes[1].axis('off')
        axes[1].set_title('Field with objects')

        # Save
        out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.png")
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)

