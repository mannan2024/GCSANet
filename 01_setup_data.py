"""Dataset setup and preprocessing for the GCSANet training pipeline."""

import glob
import os
import random
import warnings

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# These remain absolute to preserve the existing local training setup.
TRAIN_IMAGE_DIR = r"D:\PhD Data\train\image"
TRAIN_LABEL_DIR = r"D:\PhD Data\train\label"
TEST_IMAGE_DIR  = r"D:\PhD Data\test\image"
TEST_LABEL_DIR  = r"D:\PhD Data\test\label"

NUM_CLASSES = 10
TARGET_SIZE = (512, 512)
BATCH_SIZE = 8
VAL_SPLIT = 0.2
USE_AMP = True   # Mixed precision — mandatory for dual encoder on RTX 3060

CLASS_NAMES = [
    "Invalid", "Other", "Water", "Abandoned land", "Bare land",
    "Farmland", "Herbaceous", "Broadleaf forest", "Shrubland", "Orchard"
]

CLASS_COLORS = [
    [0, 0, 0], [255, 255, 255], [151, 219, 242], [38, 115, 0],
    [255, 234, 190], [245, 245, 122], [233, 255, 190], [114, 137, 68],
    [115, 178, 115], [240, 176, 207],
]

VEGETATION_INDICES = [3, 5, 6, 7, 8, 9]
VEGETATION_NAMES = ["Abandoned land", "Farmland", "Herbaceous",
                    "Broadleaf forest", "Shrubland", "Orchard"]

NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
GRAD_CLIP = 1.0
IGNORE_INDEX = 0
LABEL_SMOOTHING = 0.1

def set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

VEG_PALETTE = ["#267300", "#f5f57a", "#e9ffbe", "#728944", "#73b273", "#f0b0cf"]

set_publication_style()

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

dirs_to_create = [
    "results/models",
    "results/metrics",
    "results/figures/svg",
    "results/figures/png",
    "results/predictions/sample_predictions",
    "results/predictions/all_predictions",
    "results/ablation/ablation_1/plots",
    "results/ablation/ablation_2/plots",
    "results/ablation/ablation_3/plots",
    "results/ablation/ablation_4/plots",
    "results/ablation/ablation_summary_plots",
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)
    
print("✓ Output directory tree created:")
for d in dirs_to_create:
    print(f"  {d}/")

def rgb_to_class_index(rgb_mask: np.ndarray) -> np.ndarray:
    """Convert an RGB label mask to a class index map via exact color matching.
    
    Args:
        rgb_mask: (H, W, 3) uint8 array in RGB order.
    Returns:
        index_map: (H, W) int64 array. Unmatched pixels → 0 (Invalid).
    """
    h, w = rgb_mask.shape[:2]
    index_map = np.zeros((h, w), dtype=np.int64)
    
    for class_idx, color in enumerate(CLASS_COLORS):
        if class_idx == 0:
            continue
        match = np.all(rgb_mask == np.array(color, dtype=np.uint8), axis=-1)
        index_map[match] = class_idx
    
    return index_map


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class LULCDataset(Dataset):
    """PyTorch Dataset for LULC semantic segmentation.
    
    The mask conversion happens before resizing so class ids stay intact.
    """
    
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = sorted(image_paths)
        self.label_paths = sorted(label_paths)
        self.transform = transform
        assert len(self.image_paths) == len(self.label_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_rgb = cv2.imread(self.label_paths[idx], cv2.IMREAD_COLOR)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        
        mask = rgb_to_class_index(mask_rgb)
        
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), TARGET_SIZE,
                          interpolation=cv2.INTER_NEAREST).astype(np.int64)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        
        return img, mask.long()


print("✓ Dataset class defined with bug-proof mask pipeline")
print("  Pipeline: Load RGB → RGB→Index@full_res → Resize(bilinear/nearest) → Augment → Normalize")

train_image_paths = sorted(glob.glob(os.path.join(TRAIN_IMAGE_DIR, "*.*")))
train_label_paths = sorted(glob.glob(os.path.join(TRAIN_LABEL_DIR, "*.*")))
test_image_paths  = sorted(glob.glob(os.path.join(TEST_IMAGE_DIR, "*.*")))
test_label_paths  = sorted(glob.glob(os.path.join(TEST_LABEL_DIR, "*.*")))

print(f"Total training samples : {len(train_image_paths)}")
print(f"Total test samples     : {len(test_image_paths)}")

# ── 80/20 train/val split on filenames ──
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    train_image_paths, train_label_paths,
    test_size=VAL_SPLIT, random_state=RANDOM_SEED
)

print(f"Train split: {len(train_imgs)}, Val split: {len(val_imgs)}")

train_dataset = LULCDataset(train_imgs, train_lbls, transform=train_transform)
val_dataset   = LULCDataset(val_imgs,   val_lbls,   transform=val_transform)
test_dataset  = LULCDataset(test_image_paths, test_label_paths, transform=val_transform)

# Windows runs single-process workers here because multiprocessing can be unstable.
effective_num_workers = 0 if os.name == "nt" else NUM_WORKERS
pin_mem = DEVICE.type == "cuda"
if os.name == "nt" and NUM_WORKERS > 0:
    print(f"⚠ Windows detected: overriding NUM_WORKERS {NUM_WORKERS} -> 0 for notebook stability")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=effective_num_workers, pin_memory=pin_mem, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=effective_num_workers, pin_memory=pin_mem)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=effective_num_workers, pin_memory=pin_mem)

sample_imgs, sample_lbls = next(iter(train_loader))
print(f"\n✓ Shape verification:")
print(f"  Images : {sample_imgs.shape}  (expect [8, 3, 512, 512])")
print(f"  Labels : {sample_lbls.shape}  (expect [8, 512, 512])")
print(f"  Image dtype : {sample_imgs.dtype}, Label dtype : {sample_lbls.dtype}")
print(f"  Label range : [{sample_lbls.min().item()}, {sample_lbls.max().item()}]")

print("\n── Class distribution across training set (sampling 50 images) ──")
class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
sample_indices = np.random.choice(len(train_dataset), min(50, len(train_dataset)), replace=False)
for i in tqdm(sample_indices, desc="Counting classes"):
    _, lbl = train_dataset[i]
    for c in range(NUM_CLASSES):
        class_counts[c] += (lbl.numpy() == c).sum()

total_pixels = class_counts.sum()
print(f"\n{'Class':<20s} {'Count':>12s} {'Fraction':>10s}")
print("-" * 45)
for c in range(NUM_CLASSES):
    frac = class_counts[c] / total_pixels * 100
    flag = " ⚠ BUG?" if c == 0 and frac > 20 else ""
    print(f"{CLASS_NAMES[c]:<20s} {class_counts[c]:>12,d} {frac:>9.2f}%{flag}")

invalid_frac = class_counts[0] / total_pixels * 100
if invalid_frac > 20:
    print(f"\n🚨 CRITICAL: {invalid_frac:.1f}% Invalid pixels detected!")
    print("   This indicates the bilinear interpolation mask bug is present.")
    print("   Check the Dataset pipeline: masks MUST use INTER_NEAREST.")
else:
    print(f"\n✓ Invalid pixel fraction: {invalid_frac:.1f}% — within acceptable range")

set_publication_style()
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

for row in range(3):
    img_t, lbl_t = train_dataset[row]
    img_np = img_t.numpy().transpose(1, 2, 0) * std + mean
    img_np = np.clip(img_np, 0, 1)
    lbl_np = lbl_t.numpy()
    
    colored_mask = np.zeros((*lbl_np.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        colored_mask[lbl_np == c] = CLASS_COLORS[c]
    
    axes[row, 0].imshow(img_np)
    axes[row, 0].set_title("RGB Image")
    axes[row, 0].axis("off")
    
    axes[row, 1].imshow(colored_mask)
    axes[row, 1].set_title("Ground Truth Mask")
    axes[row, 1].axis("off")
    
    axes[row, 2].imshow(img_np)
    axes[row, 2].imshow(colored_mask, alpha=0.5)
    axes[row, 2].set_title("Overlay")
    axes[row, 2].axis("off")

legend_patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i])/255, label=CLASS_NAMES[i])
                  for i in range(NUM_CLASSES)]
fig.legend(handles=legend_patches, loc="lower center", ncol=5,
           fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.suptitle("Training Samples — Mask Encoding Verification", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
print("✓ Visual inspection: masks should show clean class boundaries (no noisy speckles)")

print("Computing class weights from full training set...")
full_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for i in tqdm(range(len(train_dataset)), desc="Scanning train set"):
    _, lbl = train_dataset[i]
    lbl_np = lbl.numpy()
    for c in range(NUM_CLASSES):
        full_class_counts[c] += (lbl_np == c).sum()

total_valid = full_class_counts[1:].sum()  # exclude Invalid
weights = np.zeros(NUM_CLASSES, dtype=np.float32)

for c in range(1, NUM_CLASSES):
    if full_class_counts[c] > 0:
        weights[c] = total_valid / (NUM_CLASSES * full_class_counts[c])
    else:
        weights[c] = 1.0

weights = np.clip(weights, 0, 10.0)
weights[0] = 0.0

CLASS_WEIGHTS = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

print(f"\n{'Class':<20s} {'Pixels':>12s} {'Weight':>8s}")
print("-" * 42)
for c in range(NUM_CLASSES):
    print(f"{CLASS_NAMES[c]:<20s} {full_class_counts[c]:>12,d} {weights[c]:>8.3f}")
print(f"\n✓ Class weights computed and moved to {DEVICE}")
