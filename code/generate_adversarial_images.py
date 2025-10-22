#!/usr/bin/env python3
"""
generate_adversarial_images.py

Robust, Mac-friendly script to:
 - pick device (MPS / CUDA / CPU)
 - load a checkpoint (with map_location)
 - run PGD attacker on one validation batch
 - save detailed 4-panel images and a flexible side-by-side grid
 - handle missing categories.txt by falling back to placeholder names
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from student_code import default_cnn_model, PGDAttack, get_val_transforms
from custom_dataloader import MiniPlacesLoader

# -------------------------
# Device selection (MPS/CUDA/CPU)
# -------------------------
if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: Apple Metal (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# -------------------------
# Model (instantiate then load weights)
# -------------------------
model = default_cnn_model(num_classes=100).to(device)

checkpoint_paths = [
    "../models/model_best.pth.tar",
    "../models/checkpoint.pth.tar",
    "../logs/simple_cnn_trained/models/model_best.pth.tar"
]
checkpoint_path = next((p for p in checkpoint_paths if os.path.exists(p)), None)
if checkpoint_path is None:
    raise FileNotFoundError(
        "No checkpoint found. Looked for:\n" + "\n".join(checkpoint_paths) +
        "\n\nPlace your model checkpoint at one of those paths or update the script."
    )

print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# support both raw state_dict and {'state_dict': ...}
state_dict = checkpoint.get("state_dict", checkpoint)

# strip module. prefix if present
cleaned_state = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        cleaned_state[k[len("module."):]] = v
    else:
        cleaned_state[k] = v

model.load_state_dict(cleaned_state)
model.eval()

# -------------------------
# Data loader
# -------------------------
val_transforms = get_val_transforms()
val_dataset = MiniPlacesLoader('../data', split='val', transforms=val_transforms)

# safe DataLoader settings for macOS / small machines
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False
)

# -------------------------
# Attacker
# -------------------------
criterion = nn.CrossEntropyLoss()
attacker = PGDAttack(criterion, num_steps=10, epsilon=8/255.0, step_size=2/255.0)

# -------------------------
# Output dir
# -------------------------
os.makedirs('../writeup/figures', exist_ok=True)

# -------------------------
# Run
# -------------------------
print("Generating adversarial examples...")

images, labels = next(iter(val_loader))
images = images.to(device)
labels = labels.to(device)

# Generate adversarial images (attacker should return images on same device/dtype)
with torch.set_grad_enabled(True):
    adv_images = attacker.perturb(model, images)

# Predictions
with torch.no_grad():
    clean_outputs = model(images)
    adv_outputs = model(adv_images)
    clean_preds = clean_outputs.argmax(dim=1)
    adv_preds = adv_outputs.argmax(dim=1)

# Denormalize (use same dtype & device as images)
mean = torch.tensor([0.485, 0.456, 0.406], dtype=images.dtype, device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], dtype=images.dtype, device=device).view(1, 3, 1, 1)

images_denorm = torch.clamp(images * std + mean, 0.0, 1.0)
adv_images_denorm = torch.clamp(adv_images * std + mean, 0.0, 1.0)

# perturbation in normalized space
perturbation = (adv_images - images).abs()

# successful attacks
success_mask = (clean_preds == labels) & (adv_preds != labels)
success_indices = torch.where(success_mask)[0].cpu()
if success_indices.numel() == 0:
    print("No successful attacks in this batch; falling back to first k images.")
    k = min(4, images.shape[0])
    success_indices = torch.arange(k)
else:
    # pick up to 8 successes (we'll visualize up to 4 detailed + up to 4 more in grid)
    success_indices = success_indices[:8]

n_examples = success_indices.numel()
print(f"\nGenerating {n_examples} examples...")

# Load categories (fallback if missing)
categories_path = '../data/categories.txt'
if os.path.exists(categories_path):
    with open(categories_path, 'r') as f:
        categories = [line.strip().split()[0] for line in f.readlines()]
else:
    print("categories.txt not found — using placeholder class names.")
    categories = [f"class_{i}" for i in range(100)]

# ------------ Detailed 4-panel comparisons (up to 4) ------------
n_detailed = min(4, n_examples)
for idx in range(n_detailed):
    i = int(success_indices[idx].item())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Clean image
    img_clean = images_denorm[i].cpu().permute(1, 2, 0).numpy()
    axes[0].imshow(img_clean)
    axes[0].set_title(f"Original: {categories[int(clean_preds[i].item())]}", fontsize=10)
    axes[0].axis('off')

    # Adversarial
    img_adv = adv_images_denorm[i].cpu().permute(1, 2, 0).numpy()
    axes[1].imshow(img_adv)
    axes[1].set_title(f"Adversarial: {categories[int(adv_preds[i].item())]}", fontsize=10)
    axes[1].axis('off')

    # Difference (amplified)
    diff = img_adv - img_clean
    diff_mag = np.abs(diff)
    # amplify and clip for display
    disp = np.clip(diff_mag * 20.0, 0.0, 1.0)
    axes[2].imshow(disp)
    axes[2].set_title("Perturbation (20x)", fontsize=10)
    axes[2].axis('off')

    # Heatmap of perturbation (mean over channels)
    pert_heat = perturbation[i].mean(0).cpu().numpy()
    vmax = float(pert_heat.max()) if pert_heat.size and pert_heat.max() > 0 else 1e-8
    im = axes[3].imshow(pert_heat, cmap='hot', vmin=0.0, vmax=vmax)
    axes[3].set_title(f"Perturbation Heatmap\nMax: {vmax:.6f}", fontsize=10)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    out_path = f"../writeup/figures/adversarial_detailed_{idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {out_path} — True={categories[int(labels[i].item())]}, "
          f"Clean={categories[int(clean_preds[i].item())]}, Adv={categories[int(adv_preds[i].item())]}")

# ------------ Flexible side-by-side grid (rows = n_examples up to 8) ------------
rows = min(8, n_examples)
if rows > 0:
    fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows))
    # If only 1 row, axes will be 1D; make it 2D for uniform indexing
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for r in range(rows):
        i = int(success_indices[r].item())
        img_clean = images_denorm[i].cpu().permute(1, 2, 0).numpy()
        img_adv = adv_images_denorm[i].cpu().permute(1, 2, 0).numpy()

        axes[r, 0].imshow(img_clean)
        axes[r, 0].set_title(f"Original: {categories[int(clean_preds[i].item())]}", fontsize=9)
        axes[r, 0].axis('off')

        axes[r, 1].imshow(img_adv)
        axes[r, 1].set_title(f"Adversarial: {categories[int(adv_preds[i].item())]}", fontsize=9)
        axes[r, 1].axis('off')

    plt.tight_layout()
    grid_path = "../writeup/figures/adversarial_sidebyside_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {grid_path}")

# ------------ Statistics ------------
n_success = int(success_mask.sum().item())
total = images.shape[0]
print("\n" + "="*50)
print("Attack Statistics:")
print("="*50)
print(f"Attack Successes (in batch): {n_success}/{total} ({100.0 * n_success / total:.1f}%)")
print(f"Clean Accuracy (batch): {(clean_preds == labels).sum().item()}/{total}")
print(f"Adversarial Accuracy (batch): {(adv_preds == labels).sum().item()}/{total}")

avg_pert = float(perturbation.mean().item())
max_pert = float(perturbation.max().item())
print("\nPerturbation Statistics:")
print(f"  Average magnitude: {avg_pert:.6f}")
print(f"  Maximum magnitude: {max_pert:.6f}")
print(f"  Epsilon bound: {8/255:.6f}")

print("\n✓ Images saved to writeup/figures/")
