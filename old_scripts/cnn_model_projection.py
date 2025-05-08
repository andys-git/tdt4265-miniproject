# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# from typing import List, Tuple
#
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from main import load_image_paths
# import random
# import matplotlib.pyplot as plt
#
#
# print("torch.__version__:", torch.__version__)
# print("CUDA available?   ", torch.cuda.is_available())
# print("CUDA version:     ", torch.version.cuda)
# print("cuDNN enabled?    ", torch.backends.cudnn.enabled)
# print("GPU count:        ", torch.cuda.device_count())
# if torch.cuda.device_count() > 0:
#     print("GPU name:         ", torch.cuda.get_device_name(0))
#
# print(torch.cuda.get_arch_list())
# print(torch.__path__)
# print(torch.version.cuda)
#
# # Hyperparameters
# IMG_WIDTH = 480
# IMG_HEIGHT = 270
# BATCH_SIZE = 8
# NUM_EPOCHS = 10
# LEARNING_RATE = 1e-5
# LAMBDA_COND = 1e-2 # strength of homography condition penalty
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# print(f"Hyperparams ▶ IMG={IMG_WIDTH}x{IMG_HEIGHT}, batch={BATCH_SIZE}, "
#       f"epochs={NUM_EPOCHS}, lr={LEARNING_RATE}, λ_cond={LAMBDA_COND}")
# print(f"Using device: {DEVICE}\n")
#
# # Dataset
# class SoccerFieldDataset(Dataset):
#     """
#     Custom dataset for edge-detected and perfect-field image pairs.
#     Expects a list of (edge_path, perfect_path) tuples.
#     """
#     def __init__(self, image_pairs: List[Tuple[str, str]]):
#         self.image_pairs = image_pairs
#
#     def __len__(self) -> int:
#         return len(self.image_pairs)
#
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         edge_path, perfect_path = self.image_pairs[idx]
#         edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
#         perfect_img = cv2.imread(perfect_path, cv2.IMREAD_GRAYSCALE)
#
#         # Resize and normalize
#         edge = cv2.resize(edge_img, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
#         perfect = cv2.resize(perfect_img, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
#
#         # Binarize
#         edge = (edge > 0.5).astype(np.float32)
#         perfect = (perfect > 0.5).astype(np.float32)
#
#         # To torch tensors: shape (1, H, W)
#         edge_tensor = torch.from_numpy(edge).unsqueeze(0)
#         perfect_tensor = torch.from_numpy(perfect).unsqueeze(0)
#         return edge_tensor, perfect_tensor
#
# # ---------------- Model Definition ----------------
# class CoordConv(nn.Module):
#     def __init__(self, with_r=False):
#         super().__init__()
#         self.with_r = with_r
#     def forward(self, x):
#         B,C,H,W = x.shape
#         ys = torch.linspace(-1,1,H,device=x.device).view(1,1,H,1).expand(B,1,H,W)
#         xs = torch.linspace(-1,1,W,device=x.device).view(1,1,1,W).expand(B,1,H,W)
#         coords = torch.cat([x, xs, ys], dim=1)
#         if self.with_r:
#             rs = torch.sqrt(xs**2 + ys**2)
#             coords = torch.cat([coords, rs], dim=1)
#         return coords
#
#
# class HomographyRegressor(nn.Module):
#     def __init__(self, H, W):
#         super().__init__()
#         self.coordconv = CoordConv(with_r=False)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, 3, stride=2, padding=1),  # now channels=3
#             nn.BatchNorm2d(32), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(),
#             nn.Conv2d(64, 128,3, stride=2, padding=1),
#             nn.BatchNorm2d(128),nn.ReLU(),
#             nn.Conv2d(128,256,3, stride=2, padding=1),
#             nn.BatchNorm2d(256),nn.ReLU(),
#         )
#         # global pooling instead of flatten
#         self.fc = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 8)
#         )
#
#
#         # init to identity‐offset
#         # nn.init.zeros_(self.fc[-1].weight)
#         # nn.init.zeros_(self.fc[-1].bias)
#         self.fc[-1].weight.data.zero_()
#         self.fc[-1].bias.data.zero_()
#
#     def forward(self, x):
#         x = self.coordconv(x)       # adds coord channels
#         feat = self.encoder(x)
#         pooled = F.adaptive_avg_pool2d(feat, 1).view(x.size(0), -1)
#         return self.fc(pooled)      # (B,8)
#
#
# # Differentiable homography warp
# def get_homography_grid(H: torch.Tensor, height: int, width: int) -> torch.Tensor:
#     """
#     Build a sampling grid for torch.nn.functional.grid_sample from a batch of 3x3 homographies.
#     H: (B,3,3), returns grid: (B, H, W, 2) in normalized coords.
#     """
#     B = H.size(0)
#     device = H.device
#     ys = torch.linspace(0, height - 1, height, device=device)
#     xs = torch.linspace(0, width  - 1, width,  device=device)
#     grid_y, grid_x = torch.meshgrid(ys, xs)
#     ones = torch.ones_like(grid_x)
#     coords = torch.stack([grid_x, grid_y, ones], dim=-1)            # (H, W, 3)
#     coords = coords.view(-1, 3).T                                  # (3, H*W)
#     coords = coords.unsqueeze(0).expand(B, -1, -1)                  # (B, 3, N)
#
#     warped = H.bmm(coords)                                         # (B, 3, N)
#     warped = warped / (warped[:, 2:3, :] + 1e-8)                    # normalize
#     warped_xy = warped[:, :2, :].view(B, 2, height, width)         # (B, 2, H, W)
#
#     # Normalize to [-1,1]
#     warped_xy[:, 0, :, :] = warped_xy[:, 0, :, :] / (width  - 1) * 2 - 1
#     warped_xy[:, 1, :, :] = warped_xy[:, 1, :, :] / (height - 1) * 2 - 1
#     return warped_xy.permute(0, 2, 3, 1)                            # (B, H, W, 2)
#
# # Warping Utility
# def warp_image(img: np.ndarray, H: np.ndarray) -> np.ndarray:
#     # Warp a single-channel image using a homography matrix H.
#     h, w = img.shape
#     return cv2.warpPerspective(img, H, (w, h))
#
# def warp_batch(img: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
#     # img: (B,1,H,W), H: (B,3,3)
#     grid = get_homography_grid(H, img.size(2), img.size(3))
#     return F.grid_sample(img, grid, align_corners=True)
#
# # Loss Function
# def self_supervised_loss(
#     preds: torch.Tensor,
#     edges: torch.Tensor,
#     perfects: torch.Tensor,
#     lambda_cond: float = LAMBDA_COND,
#     alpha_fp:   float = 1.0   # weight for false‐positive penalty
# ) -> torch.Tensor:
#     B,_,H,W = perfects.shape
#     device = preds.device
#
#     # 1) build H
#     id8    = torch.tensor([1,0,0,0,1,0,0,0], device=device).unsqueeze(0).expand(B,-1)
#     H8     = preds + id8
#     H_full = torch.cat([H8, torch.ones((B,1),device=device)],dim=1).view(B,3,3)
#
#     # 2) warp
#     warped      = warp_batch(perfects, H_full).clamp(0,1)    # (B,1,H,W)
#     valid_mask  = warp_batch(torch.ones_like(perfects), H_full).clamp(0,1)
#
#     w_pos, w_neg = 1.0, 5.0  # tune w_neg >> w_pos
#     weights = edges * w_pos + (1.0 - edges) * w_neg
#
#     # 3) BCE on valid pixels (as before)
#     bce_map = F.binary_cross_entropy(warped, edges, weight=weights, reduction='none')
#     L_bce = (bce_map * valid_mask).sum() / (valid_mask.sum() + 1e-6)
#
#     # 4) False‐positive penalty
#     #   any warped==1 where edges==0 is a false positive
#     nonedge_mask = (1.0 - edges) * valid_mask            # (B,1,H,W)
#     fp_loss      = (warped * nonedge_mask).mean()        # fraction of non‐edge pixels turned on
#     L_fp         = alpha_fp * fp_loss.pow(2)
#
#     # 5) Condition‐number penalty (as before)
#     U, S, V = torch.svd(H_full)
#     cond    = S[:,0]/(S[:,-1] + 1e-8)
#     L_cond  = lambda_cond * ((cond - 1.0)**2).mean()
#
#     return L_bce + L_fp + L_cond
#
#
#
# # Training loop
# def train(model: nn.Module,
#           dataloader: DataLoader,
#           num_epochs: int,
#           lr: float,
#           save_path: str):
#     print(f"Starting training on {DEVICE} with {len(dataloader)} batches per epoch\n")
#
#     # ——— 1) Build your 8-element offset vector on CPU
#     H_init = np.array([
#         [ 1.59389586e+00, -5.39204121e-01, -1.39000000e+02],
#         [-9.55232264e-03,  2.08084109e-01,  5.20000000e+01],
#         [-6.73943396e-05, -2.20864138e-03,  9.99999999e-01]
#     ], dtype=np.float32)
#     offsets = torch.tensor([
#         H_init[0,0] - 1.0, H_init[0,1], H_init[0,2],
#         H_init[1,0],      H_init[1,1] - 1.0, H_init[1,2],
#         H_init[2,0],      H_init[2,1]
#     ], dtype=torch.float32)
#
#     # ——— 2) Zero the final-layer weights, copy your offsets into its bias
#     model.fc[-1].weight.data.zero_()
#     model.fc[-1].bias.data.copy_(offsets.to(model.fc[-1].bias.device))
#
#     # ——— 3) Send model to DEVICE *before* creating optimizer
#     model.to(DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     # (Optional) Small supervised warm-up on MSE to anchor the network
#     # for edges,perfects in supervised_loader:
#     #     optimizer.zero_grad()
#     #     pred8 = model(edges.to(DEVICE))
#     #     target = offsets.unsqueeze(0).expand_as(pred8).to(DEVICE)
#     #     loss = F.mse_loss(pred8, target)
#     #     loss.backward()
#     #     optimizer.step()
#
#     # ——— 4) Now your regular self-supervised loop
#     for epoch in range(1, num_epochs + 1):
#         print(f"--- Epoch {epoch}/{num_epochs} ---")
#         model.train()
#         epoch_loss = 0.0
#         for i, (edges, perfects) in enumerate(dataloader):
#             edges    = edges.to(DEVICE)
#             perfects = perfects.to(DEVICE)
#
#             optimizer.zero_grad()
#             preds = model(edges)                           # (B,8)
#             loss  = self_supervised_loss(preds, edges, perfects)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item() * edges.size(0)
#             if i % 10 == 0:
#                 print(f"  batch {i:3d}/{len(dataloader):3d} — loss: {loss.item():.4f}")
#
#         avg_loss = epoch_loss / len(dataloader.dataset)
#         print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")
#
#     # ——— 5) Save final weights
#     torch.save(model.state_dict(), save_path)
#     print(f"Model saved to {save_path}")
#
#
# def visualize_warp_on_source(sample_pair, model):
#     """
#     sample_pair: tuple (edge_path, perfect_path)
#     model: your trained HomographyRegressor
#     """
#     edge_path, perfect_path = sample_pair
#
#     # 1) load & resize source image
#     source_path = edge_path.replace('_e', '')   # adjust to your naming
#     source_img  = cv2.imread(source_path)
#     source_img  = cv2.resize(source_img, (IMG_WIDTH, IMG_HEIGHT))
#
#     # 2) load & binarize perfect-field
#     perfect = cv2.imread(perfect_path, cv2.IMREAD_GRAYSCALE)
#     perfect = cv2.resize(perfect, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
#     perfect_bin = (perfect > 0.5).astype(np.float32)
#
#     # 3) load & binarize edge image
#     edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
#     edge = cv2.resize(edge, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
#     edge_bin = (edge > 0.5).astype(np.float32)
#
#     # prepare tensors
#     edge_t    = torch.from_numpy(edge_bin).unsqueeze(0).unsqueeze(0).to(DEVICE)
#     perfect_t = torch.from_numpy(perfect_bin).unsqueeze(0).unsqueeze(0).to(DEVICE)
#
#     # 4) predict homography
#     model.eval()
#     with torch.no_grad():
#         preds = model(edge_t)  # (1,8)
#         id8   = torch.tensor([1,0,0,0,1,0,0,0], device=DEVICE).unsqueeze(0)
#         H8    = preds + id8
#         H_full= torch.cat([H8, torch.ones((1,1), device=DEVICE)], dim=1)\
#                     .view(1,3,3).cpu().numpy()[0]
#
#     # 5) warp the perfect mask
#     warped     = warp_image(perfect_bin, H_full)      # floats [0..1]
#     warped_bin = (warped > 0.5).astype(np.uint8)      # 0 or 1
#
#     # 6) overlay cyan on source
#     overlay = source_img.copy()
#     overlay[warped_bin == 1] = (255, 255, 0)  # BGR cyan
#     overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
#
#     # 7) plot side by side
#     edge_rgb = cv2.cvtColor((edge_bin*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     ax1.imshow(edge_rgb)
#     ax1.set_title("Input Edge Map")
#     ax1.axis('off')
#
#     ax2.imshow(overlay_rgb)
#     ax2.set_title("Source with Warped Lines (cyan)")
#     ax2.axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     paths = [
#         "RBK_TDT17/1_train-val_1min_aalesund_from_start/img1_e",
#         "RBK_TDT17/2_train-val_1min_after_goal/img1_e",
#         "RBK_TDT17/3_test_1min_hamkam_from_start/img1_e",
#         "RBK_TDT17/4_annotate_1min_bodo_start/img1_e"
#     ]
#
#     image_pairs = []
#
#     for path in paths:
#         loaded_paths = load_image_paths(path)
#         for loaded_path in loaded_paths:
#             image_pairs.append((loaded_path, 'field.jpg'))
#
#     # edge_img = cv2.imread(image_pairs[0][0], cv2.IMREAD_GRAYSCALE)
#     # perfect_img = cv2.imread(image_pairs[0][1], cv2.IMREAD_GRAYSCALE)
#
#     # print("edge_img dtype:", edge_img.dtype, "min/max:", edge_img.min(), edge_img.max())
#     # print("unique edge_img values:", np.unique(edge_img)[:10], "…")
#     # print("perfect_img dtype:", perfect_img.dtype, "min/max:", perfect_img.min(), perfect_img.max())
#     # print("unique perfect_img values:", np.unique(perfect_img)[:10], "…\n")
#     #
#     # display_image(edge_img)
#     # display_image(perfect_img)
#
#     ## --- MODEL UNCOMMENT TO TRAIN ---
#     dataset = SoccerFieldDataset(image_pairs)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
#
#     model = HomographyRegressor(IMG_HEIGHT, IMG_WIDTH)
#     train(model, loader, NUM_EPOCHS, LEARNING_RATE, save_path="homography_model.pth")
#
#     ## --- VISUALIZE UNCOMMENT TO SEE ---
#     model = HomographyRegressor(IMG_HEIGHT, IMG_WIDTH)
#     state = torch.load("homography_model.pth", map_location=DEVICE)
#     model.load_state_dict(state)
#     model.to(DEVICE)
#     model.eval()
#
#     for _ in range(5):
#         example = random.choice(image_pairs)
#         visualize_warp_on_source(example, model)
