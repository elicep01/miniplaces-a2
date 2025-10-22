import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image
import os

from student_code import default_cnn_model, PGDAttack, get_val_transforms
from custom_dataloader import MiniPlacesLoader

# Setup
device = torch.device("cuda:0")
model = default_cnn_model(num_classes=100).to(device)

# Load trained model
checkpoint = torch.load('../logs/simple_cnn_trained/models/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load data
val_transforms = get_val_transforms()
val_dataset = MiniPlacesLoader(
    '../data', split='val', transforms=val_transforms
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, shuffle=True, num_workers=2
)

# Create PGD attacker
criterion = nn.CrossEntropyLoss()
attacker = PGDAttack(criterion, num_steps=10, epsilon=8/255.0, step_size=2/255.0)

# Create output directory
os.makedirs('../writeup/figures', exist_ok=True)

print("Generating adversarial examples...")

# Get one batch
images, labels = next(iter(val_loader))
images = images.to(device)
labels = labels.to(device)

# Generate adversarial examples
with torch.set_grad_enabled(True):
    adv_images = attacker.perturb(model, images)

# Get predictions
with torch.no_grad():
    clean_outputs = model(images)
    adv_outputs = model(adv_images)
    
    clean_preds = clean_outputs.argmax(dim=1)
    adv_preds = adv_outputs.argmax(dim=1)

# Denormalize for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

def denormalize(img):
    return img * std + mean

images_denorm = denormalize(images)
adv_images_denorm = denormalize(adv_images)

# Create difference map (amplified for visibility)
diff = torch.abs(adv_images_denorm - images_denorm)
diff_amplified = torch.clamp(diff * 10, 0, 1)

# Save individual examples
print("Saving comparison images...")

# Select 8 interesting examples (where attack succeeded)
success_mask = (clean_preds == labels) & (adv_preds != labels)
success_indices = torch.where(success_mask)[0][:8]

if len(success_indices) < 8:
    success_indices = torch.arange(min(8, len(labels)))

for idx, i in enumerate(success_indices):
    # Create comparison grid for this image
    comparison = torch.stack([
        images_denorm[i],
        adv_images_denorm[i],
        diff_amplified[i]
    ])
    
    save_image(comparison, f'../writeup/figures/adversarial_example_{idx}.png', nrow=3, padding=2)
    
    print(f"Example {idx}: True={labels[i].item()}, "
          f"Clean Pred={clean_preds[i].item()}, "
          f"Adv Pred={adv_preds[i].item()}")

# Create big comparison grid
print("Creating comparison grid...")
n_show = 8
grid_images = torch.cat([
    images_denorm[:n_show],
    adv_images_denorm[:n_show],
    diff_amplified[:n_show]
], dim=0)

save_image(grid_images, '../writeup/figures/adversarial_comparison_grid.png', 
           nrow=n_show, padding=5, pad_value=1.0)

# Create side-by-side for paper
print("Creating side-by-side comparison...")
for i in range(4):
    side_by_side = torch.cat([
        images_denorm[i:i+1],
        adv_images_denorm[i:i+1]
    ], dim=0)
    save_image(side_by_side, f'../writeup/figures/sidebyside_{i}.png', 
               nrow=2, padding=2)

print("\nDone! Images saved to writeup/figures/")
print("\nFiles created:")
print("  - adversarial_example_0-7.png (individual comparisons)")
print("  - adversarial_comparison_grid.png (big grid)")
print("  - sidebyside_0-3.png (clean vs adversarial pairs)")
