# Google Colab Training Guide — FlagshipNet
# ============================================
# Copy these cells into a Colab notebook (File → New Notebook)
# Use Runtime → Change runtime type → T4 GPU
#
# STEPS:
# 1. Upload the entire model2_oc/ folder to your Google Drive
# 2. Upload the data/ folder to your Google Drive
# 3. Copy each cell below into a Colab notebook cell
# 4. Run cells in order

# ============================================================
# CELL 1: Setup & Mount Drive
# ============================================================
# !pip install torch torchvision pillow matplotlib numpy

from google.colab import drive
drive.mount('/content/drive')

# Update this path to where you uploaded model2_oc/ in your Drive
PROJECT_DIR = '/content/drive/MyDrive/Semicon-Hackathon-KLA-Solution/model2_oc'
DATA_DIR = '/content/drive/MyDrive/Semicon-Hackathon-KLA-Solution/data'

import os
assert os.path.isdir(PROJECT_DIR), f"Project dir not found: {PROJECT_DIR}"
assert os.path.isdir(DATA_DIR), f"Data dir not found: {DATA_DIR}"
print(f"Project: {PROJECT_DIR}")
print(f"Data: {DATA_DIR}")

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: No GPU found. Training will be very slow.")
    print("Go to Runtime → Change runtime type → T4 GPU")


# ============================================================
# CELL 2: Quick Sanity Check
# ============================================================
import sys
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

from models.flagship import FlagshipNet, count_params
import torch

model = FlagshipNet()
params = count_params(model)
print(f"FlagshipNet params: {params:,} ({params/1e6:.2f}M)")

# Quick forward pass
x = torch.randn(2, 1, 128, 128).cuda()
model = model.cuda()
with torch.no_grad():
    clean, hr = model.forward_two_stage(x)
print(f"Input:  {x.shape}")
print(f"Clean:  {clean.shape}")
print(f"HR:     {hr.shape}")
print("Sanity check PASSED")


# ============================================================
# CELL 3: Train FlagshipNet (200 epochs)
# ============================================================
# !cd {PROJECT_DIR} && python scripts/train_flagship.py \
#     --data_dir {DATA_DIR} \
#     --output_dir {PROJECT_DIR}/../weights \
#     --epochs 200 \
#     --batch_size 16 \
#     --lr 2e-4 \
#     --save_every 25

# --- OR run this Python equivalent for more control ---

import sys, os, time, json
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))
from models.flagship import FlagshipNet, build_flagship, count_params
from losses.flagship_loss import FlagshipLoss
from data.dataset import KLADataset
from utils.metrics import compute_ssim, compute_psnr
from utils.guards import clip_output

# Config
EPOCHS = 200
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 42
OUTPUT_DIR = os.path.join(PROJECT_DIR, '..', 'weights')

# Seed
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda')
print(f"Device: {device}")

# Data
train_ds = KLADataset(DATA_DIR, split='train', synthetic_aug=True)
val_ds = KLADataset(DATA_DIR, split='val', synthetic_aug=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

# Model
model = build_flagship().to(device)
params = count_params(model)
print(f"Model: {params:,} params ({params/1e6:.2f}M)")

# Loss & Optimizer
loss_fn = FlagshipLoss(total_epochs=EPOCHS).to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda')

print(f"Training {EPOCHS} epochs, lr={LR}")
print("-" * 70)

# Validate function
@torch.no_grad()
def validate(model, loader, device, n_limit=200):
    model.eval()
    ssims, psnrs = [], []
    cnt = 0
    for lr_b, gt_b in loader:
        lr_b, gt_b = lr_b.to(device), gt_b.to(device)
        _, out = model.forward_two_stage(lr_b)
        out = torch.nan_to_num(out.clamp(0.0, 1.0))
        out_np = out.squeeze(1).cpu().numpy()
        gt_np = gt_b.clamp(0.0, 1.0).squeeze(1).cpu().numpy()
        for i in range(out_np.shape[0]):
            ssims.append(compute_ssim(out_np[i], gt_np[i]))
            psnrs.append(compute_psnr(out_np[i], gt_np[i]))
            cnt += 1
            if cnt >= n_limit:
                break
        if cnt >= n_limit:
            break
    return float(np.mean(ssims)), float(np.mean(psnrs))


# Training loop
history = []
best_ssim = -1.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    loss_fn.set_epoch(epoch)
    running = 0.0
    seen = 0

    for lr_b, gt_b in train_loader:
        lr_b, gt_b = lr_b.to(device), gt_b.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            denoised, hr = model.forward_two_stage(lr_b)
            total, ldict = loss_fn(hr, gt_b, denoised, lr_b)

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running += total.item() * lr_b.size(0)
        seen += lr_b.size(0)

    scheduler.step()
    train_loss = running / max(seen, 1)
    val_ssim, val_psnr = validate(model, val_loader, device)
    elapsed = time.time() - t0
    print(f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} | "
          f"val_ssim={val_ssim:.4f} | val_psnr={val_psnr:.2f} | {elapsed:.1f}s")

    history.append({"epoch": epoch, "loss": train_loss,
                    "val_ssim": val_ssim, "val_psnr": val_psnr})

    if val_ssim > best_ssim:
        best_ssim = val_ssim
        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "val_ssim": val_ssim, "val_psnr": val_psnr,
            "model_type": "flagship", "cfg": {},
        }
        torch.save(ckpt, os.path.join(OUTPUT_DIR, 'best_model.pt'))
        print(f"  -> New best SSIM: {val_ssim:.4f}")

    if epoch % 25 == 0:
        torch.save({"epoch": epoch, "model_state": model.state_dict()},
                   os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pt'))

with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
    json.dump(history, f, indent=2)
print(f"\nDone. Best SSIM: {best_ssim:.4f}")


# ============================================================
# CELL 4: Evaluate & Compare
# ============================================================
# Load best weights and run on validation set
model = build_flagship().to(device)
ckpt = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt'),
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Loaded best model from epoch {ckpt.get('epoch', '?')} "
      f"(SSIM={ckpt.get('val_ssim', '?'):.4f})")

# Run benchmark
val_ds = KLADataset(DATA_DIR, split='val', synthetic_aug=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

all_ssims, all_psnrs = [], []
with torch.no_grad():
    for i, (lr_b, gt_b) in enumerate(val_loader):
        lr_b, gt_b = lr_b.to(device), gt_b.to(device)
        _, out = model.forward_two_stage(lr_b)
        out_np = out.squeeze().cpu().numpy()
        gt_np = gt_b.squeeze().cpu().numpy()
        all_ssims.append(compute_ssim(out_np, gt_np))
        all_psnrs.append(compute_psnr(out_np, gt_np))
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(val_loader)}] "
                  f"SSIM={np.mean(all_ssims):.4f} PSNR={np.mean(all_psnrs):.2f}")

print(f"\nFinal: SSIM={np.mean(all_ssims):.4f} ± {np.std(all_ssims):.4f}")
print(f"       PSNR={np.mean(all_psnrs):.2f} ± {np.std(all_psnrs):.2f}")


# ============================================================
# CELL 5: Visualize Results
# ============================================================
import matplotlib.pyplot as plt

model.eval()
fig, axes = plt.subplots(5, 3, figsize=(15, 25))
samples = [0, 50, 100, 200, 400]

for row, idx in enumerate(samples):
    lr_b, gt_b = val_ds[idx]
    lr_b = lr_b.unsqueeze(0).to(device)

    with torch.no_grad():
        _, out = model.forward_two_stage(lr_b)

    lr_img = lr_b.squeeze().cpu().numpy()
    out_img = out.squeeze().cpu().numpy()
    gt_img = gt_b.squeeze().numpy()

    axes[row, 0].imshow(lr_img, cmap='gray', vmin=0, vmax=1)
    axes[row, 0].set_title(f'LR Input ({lr_img.shape})')
    axes[row, 0].axis('off')

    axes[row, 1].imshow(out_img, cmap='gray', vmin=0, vmax=1)
    ssim_val = compute_ssim(out_img, gt_img)
    axes[row, 1].set_title(f'Restored (SSIM={ssim_val:.4f})')
    axes[row, 1].axis('off')

    axes[row, 2].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
    axes[row, 2].set_title(f'Ground Truth ({gt_img.shape})')
    axes[row, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'results_visualization.png'), dpi=150)
plt.show()
print(f"Saved visualization to {OUTPUT_DIR}/results_visualization.png")


# ============================================================
# CELL 6: Export to ONNX (for deployment)
# ============================================================
model_cpu = build_flagship()
model_cpu.load_state_dict(ckpt['model_state'])
model_cpu.eval()

dummy = torch.randn(1, 1, 128, 128)
onnx_path = os.path.join(OUTPUT_DIR, 'flagshipnet.onnx')
torch.onnx.export(
    model_cpu, dummy, onnx_path,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17,
)
print(f"Exported ONNX: {onnx_path}")
print(f"File size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")
