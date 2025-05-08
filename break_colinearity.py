import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_detections_and_synthetics(image_path, detections, synthetics):
    """
    Plots an image with overlaid original detection points and synthetic points.

    Args:
      image_path (str):
        Path to the image file.
      detections (list of tuples):
        [(name, confidence, x, y), …] for your original detected points.
      synthetics (list of tuples):
        [(name, confidence, x, y), …] for the injected synthetic points.

    Usage:
      dets = frame_dict[frame]['points']
      syns = inject_synthetic_points(frame, frame_dict, img_folder)
      plot_detections_and_synthetics(os.path.join(img_folder, frame), dets, syns)
    """
    # load & convert
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.axis('off')

    # plot originals in blue circles
    for name, conf, x, y in detections:
        plt.scatter(x, y, marker='o', edgecolors='white', facecolors='none', s=100, lw=2)
        plt.text(x+5, y+5, name, color='white', fontsize=9, weight='bold')

    # plot synthetics in red crosses
    for name, conf, x, y in synthetics:
        plt.scatter(x, y, marker='x', s=100, lw=2, color='red')
        plt.text(x+5, y+5, name, color='red', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.show()

def detect_edges(image):
    h_img, w_img = image.shape[:2]

    # mask green + boost white + edges
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, (25,40,60), (95,255,255))
    masked = cv2.bitwise_and(image, image, mask=hsv_mask)

    lab2 = cv2.cvtColor(masked, cv2.COLOR_BGR2Lab)
    l_channel = lab2[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l_channel)

    _, white_mask = cv2.threshold(l_eq, 150, 255, cv2.THRESH_BINARY)
    edges = cv2.dilate(
        white_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),
        iterations=3
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    min_area = 800
    edges_clean = np.zeros_like(white_mask)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            edges_clean[labels == lab] = 255

    # downsample for speed
    edges_clean = cv2.resize(edges_clean, (w_img // 4, h_img // 4), interpolation=cv2.INTER_AREA)
    _, edges_clean = cv2.threshold(edges_clean, 127, 255, cv2.THRESH_BINARY)

    return edges_clean

def inject_synthetic_points(frame_name, frame_dict, image_folder):
    # 1) pull out detections
    pts = frame_dict.get(frame_name, {}).get('points', [])
    required = {'pitch_top_middle','top_central_circle','middle_central_circle','bottom_central_circle'}
    if {p[0] for p in pts} != required:
        return []
    coords = {name: np.array([x, y], float) for name,_,x,y in pts}
    top_cc, mid_cc, bot_cc = coords['top_central_circle'], coords['middle_central_circle'], coords['bottom_central_circle']

    # 2) load image & edge detect
    img = cv2.imread(os.path.join(image_folder, frame_name))
    if img is None:
        raise FileNotFoundError(frame_name)
    edges = detect_edges(img)
    h2, w2 = edges.shape

    # 3) compute minor-axis exactly from top/bottom
    scale = 4
    # full small‐scale diameter from top to bottom
    D_small = abs(bot_cc[1] - top_cc[1]) / scale
    ry_small = int(round(D_small / 2.0))
    # center in small‐coords:   x from mid, y from midpoint of top/bottom
    cx_small = int(round(mid_cc[0] / scale))
    cy_small = int(round((top_cc[1] + bot_cc[1]) / (2.0 * scale)))
    center_small = (cx_small, cy_small)

    # sweep limit for rx
    max_rx = min(cx_small, w2 - cx_small - 1)

    # restrict ROI to just around the ellipse
    x1 = max(0, cx_small - max_rx)
    x2 = min(w2, cx_small + max_rx + 1)
    y1 = max(0, cy_small - ry_small)
    y2 = min(h2, cy_small + ry_small + 1)
    edges_roi = edges[y1:y2, x1:x2]

    # center in ROI coords
    cx_roi = cx_small - x1
    cy_roi = cy_small - y1

    best_rx, best_score = 0, -1
    for rx in range(1, max_rx+1):
        mask = np.zeros_like(edges_roi, np.uint8)
        cv2.ellipse(mask,
                    (cx_roi, cy_roi),
                    (rx, ry_small),
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=255,
                    thickness=1)
        score = cv2.countNonZero(cv2.bitwise_and(mask, edges_roi))
        if score > best_score:
            best_score, best_rx = score, rx

    # ### DEBUG PRINT
    # # 1) convert edges to RGB so we can draw a colored ellipse
    # edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    #
    # # 2) draw the best-fit ellipse in red
    # cv2.ellipse(edges_rgb,
    #             center_small,
    #             (best_rx, ry_small),
    #             angle=0,
    #             startAngle=0,
    #             endAngle=360,
    #             color=(0, 0, 255),  # red in BGR
    #             thickness=2)
    #
    # # 3) mark the center in green
    # cv2.circle(edges_rgb,
    #            center_small,
    #            radius=3,
    #            color=(0, 255, 0),  # green in BGR
    #            thickness=-1)
    #
    # # 4) display with matplotlib
    # plt.figure(figsize=(6, 6))
    # plt.imshow(cv2.cvtColor(edges_rgb, cv2.COLOR_BGR2RGB))
    # plt.title(f"Best ellipse: rx={best_rx}, ry={ry_small}")
    # plt.axis('off')
    # plt.show()

    # convert back to full-res
    r_x = best_rx * scale
    r_y = D_small * scale / 2.0  # (should equal abs(mid_cc[1]-top_cc[1]))

    # 4) define axis & perp (unchanged)
    axis = bot_cc - top_cc
    axis_norm = axis / np.linalg.norm(axis)
    perp = np.array([ axis_norm[1], -axis_norm[0] ])

    # synthetic points
    syn_right = mid_cc + perp * r_x
    s45 = np.sqrt(2)/2
    syn_tl = mid_cc + (-perp * (r_x * s45) ) + (-axis_norm * (r_y * s45))

    out = [
        ('synthetic_central_circle_right',    1.0, float(syn_right[0]),  float(syn_right[1])),
        ('synthetic_central_circle_topleft', 1.0, float(syn_tl[0]),     float(syn_tl[1]))
    ]

    # # debug‐plot
    # plot_detections_and_synthetics(
    #     os.path.join(image_folder, frame_name),
    #     pts,
    #     out
    # )
    return out
