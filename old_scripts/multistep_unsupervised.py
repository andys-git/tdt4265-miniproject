import os
import glob
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------- Configuration -----------------
IMAGE_ROOT    = "./data"   # root folder containing subfolders img1 and img1_e
PERFECT_FIELD = "field.jpg"
OUTPUT_MODEL  = "homography_final.pth"

IMG_W, IMG_H = 480, 270
BATCH_SIZE   = 8
LR           = 1e-4
EPOCHS       = 20
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# ----------------- Helpers -----------------
def list_image_pairs():
    paths = [
        "RBK_TDT17/1_train-val_1min_aalesund_from_start/",
        "RBK_TDT17/2_train-val_1min_after_goal/",
        "RBK_TDT17/3_test_1min_hamkam_from_start/",
        "RBK_TDT17/4_annotate_1min_bodo_start/"
    ]

    pairs = []

    for path in paths:
        for edge_path in glob.glob(os.path.join(path, "img1_e/*.jpg")):
            color_path = edge_path.replace("_e", "")
            if os.path.exists(color_path):
                pairs.append((edge_path, color_path))

    return pairs

# ----------------- Unsupervised Cues -----------------
def detect_center_circle(edge_img):
    # Hough circle on binary edge map
    circles = cv2.HoughCircles(edge_img, cv2.HOUGH_GRADIENT,
                               dp=1.2, minDist=edge_img.shape[0]/4,
                               param1=50, param2=30,
                               minRadius=10, maxRadius=100)
    if circles is None:
        # indicate no detection
        return (None, None), 0.0
    x, y, r = circles[0,0]
    conf = np.clip(r / 100.0, 0.0, 1.0)
    return (x, y), conf
    x, y, r = circles[0,0]
    conf = np.clip(r / 100.0, 0.0, 1.0)
    return (x, y), conf


def find_field_corners(color_img):
    # isolate green via HSV
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    eps = 0.02 * cv2.arcLength(cnt, True)
    quad = cv2.approxPolyDP(cnt, eps, True)
    if len(quad) == 4:
        corners = quad.reshape(4,2).astype(float)
    else:
        pts = cnt.reshape(-1,2)
        xs, ys = pts[:,0], pts[:,1]
        corners = np.array([
            [xs.min(), ys.min()], [xs.max(), ys.min()],
            [xs.max(), ys.max()], [xs.min(), ys.max()]
        ], float)
    # sort TL, TR, BR, BL
    sorted_y = corners[np.argsort(corners[:,1])]
    top2 = sorted_y[:2][np.argsort(sorted_y[:2,0])]
    bot2 = sorted_y[2:][np.argsort(sorted_y[2:,0])]
    return np.vstack([top2, bot2])

# Dataset
class HomographyCueDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        epath, cpath = self.pairs[idx]
        # load and resize
        edge = cv2.imread(epath, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(cpath)
        edge = cv2.resize(edge, (IMG_W, IMG_H))
        color = cv2.resize(color, (IMG_W, IMG_H))
        # binarize
        _, edge_bin = cv2.threshold(edge, 128, 1, cv2.THRESH_BINARY)
        # circle cue
        (cx, cy), conf = detect_center_circle(edge)
        circle_map = np.zeros((IMG_H, IMG_W), np.float32)
        if cx is not None:
            conf_val = float(conf)
            px = int(cx * IMG_W / edge.shape[1])
            py = int(cy * IMG_H / edge.shape[0])
            cv2.circle(circle_map, (px, py), 5, conf_val, -1)
        # green mask for corners
        corners = find_field_corners(color)
        # if no corners found, use image-corner defaults
        if corners is None:
            corners = np.array([[0,0], [IMG_W-1,0], [IMG_W-1,IMG_H-1], [0,IMG_H-1]], float)
        # build target vector shape (8,)
        target = corners.astype(np.float32).reshape(-1)  # [x0,y0,...,x3,y3]
                # green isolation
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35,50,50), (85,255,255)) / 255.0

        # corner_map: heatmap of detected corners
        corner_map = np.zeros((IMG_H, IMG_W), np.float32)
        for x, y in corners:
            xi = int(x * IMG_W / color.shape[1])
            yi = int(y * IMG_H / color.shape[0])
            cv2.circle(corner_map, (xi, yi), 5, 1.0, -1)

        # stack into input tensor (4 channels)
        inp = np.stack([edge_bin, circle_map, corner_map, green_mask], axis=0).astype(np.float32)
        # return input and target coords
        return torch.from_numpy(inp), torch.from_numpy(target)

# Final Model
class FinalRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,8)
        )
    def forward(self, x):
        f = self.encoder(x).view(x.size(0),-1)
        return self.fc(f)

# Loss
def corner_loss(preds, targets, margin=1.0):
    # preds, targets: (B,8) two x4 coords
    Lmse = F.mse_loss(preds, targets)
    # ordering penalty: x0<x1<x2<x3, y0<y2 etc
    # here we skip detailed hinge for brevity
    return Lmse

# Train
def train():
    pairs = list_image_pairs()
    ds = HomographyCueDataset(pairs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = FinalRegressor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        epoch_loss = 0.0
        for inp, target in loader:
            inp = inp.to(DEVICE)
            target = target.to(DEVICE)                    # use actual corner targets
            preds = model(inp)                            # shape (B,8)
            loss = corner_loss(preds, target)             # compute loss against true corners
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * inp.size(0)
        avg_loss = epoch_loss / len(ds)
        print(f"Epoch {ep+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Model saved to {OUTPUT_MODEL}")

# ----------------- Visualization -----------------
def visualize(model, example_pair):
    model.eval()
    with torch.no_grad():
        inp, _ = HomographyCueDataset([example_pair])[0]
        pred = model(inp.unsqueeze(0).to(DEVICE)).cpu().view(4,2).numpy()
    # show on color image
    _, cpath = example_pair
    img = cv2.resize(cv2.imread(cpath), (IMG_W, IMG_H))
    for x,y in pred:
        cv2.circle(img, (int(x),int(y)), 5, (0,0,255), -1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.show()

# ----------------- Main -----------------
if __name__ == '__main__':
    # train()
    # example visualize
    pairs = list_image_pairs()
    model = FinalRegressor().to(DEVICE)
    model.load_state_dict(torch.load(OUTPUT_MODEL))
    visualize(model, random.choice(pairs))
    list_image_pairs()
