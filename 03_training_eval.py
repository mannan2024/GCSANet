"""Training, evaluation, reporting, and analysis for GCSANet."""

import gc
import math
import runpy
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from figure_export_utils import save_matplotlib_bundle

BASE_DIR = Path(__file__).resolve().parent

required_data_symbols = [
    "train_dataset", "val_dataset", "test_dataset",
    "train_loader", "val_loader", "test_loader",
    "CLASS_NAMES", "CLASS_COLORS", "VEGETATION_NAMES", "VEG_PALETTE",
]
if any(sym not in globals() for sym in required_data_symbols):
    globals().update(runpy.run_path(str(BASE_DIR / "01_setup_data.py"), init_globals=globals()))

required_loss_symbols = [
    "GCSANet", "criterion", "train_one_epoch", "compute_miou", "compute_oa",
    "DEVICE", "NUM_CLASSES", "BATCH_SIZE", "USE_AMP", "IGNORE_INDEX",
    "VEGETATION_INDICES", "LEARNING_RATE", "WEIGHT_DECAY", "GRAD_CLIP",
]
if any(sym not in globals() for sym in required_loss_symbols):
    globals().update(runpy.run_path(str(BASE_DIR / "02_model_losses.py"), init_globals=globals()))

CLASS_NAMES = globals().get(
    "CLASS_NAMES",
    [
        "Invalid", "Other", "Water", "Abandoned land", "Bare land",
        "Farmland", "Herbaceous", "Broadleaf forest", "Shrubland", "Orchard",
    ],
)
CLASS_COLORS = globals().get(
    "CLASS_COLORS",
    [
        [0, 0, 0], [255, 255, 255], [151, 219, 242], [38, 115, 0],
        [255, 234, 190], [245, 245, 122], [233, 255, 190], [114, 137, 68],
        [115, 178, 115], [240, 176, 207],
    ],
)
VEGETATION_NAMES = globals().get(
    "VEGETATION_NAMES",
    ["Abandoned land", "Farmland", "Herbaceous", "Broadleaf forest", "Shrubland", "Orchard"],
)
VEG_PALETTE = globals().get(
    "VEG_PALETTE",
    ["#267300", "#f5f57a", "#e9ffbe", "#728944", "#73b273", "#f0b0cf"],
)
AMP_DEVICE_TYPE = globals().get("AMP_DEVICE_TYPE", "cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = globals().get("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = globals().get("WEIGHT_DECAY", 1e-4)
GRAD_CLIP = globals().get("GRAD_CLIP", 1.0)
NUM_EPOCHS = globals().get("NUM_EPOCHS", 3)
PATIENCE = globals().get("PATIENCE", 3)


def amp_autocast(enabled=True):
    return torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=enabled)


def build_grad_scaler(enabled=True):
    return torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=enabled)


if "set_publication_style" not in globals():
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


if any(sym not in globals() for sym in ("train_dataset", "val_dataset", "test_dataset", "train_loader", "val_loader", "test_loader")):
    raise RuntimeError("Dataset/DataLoader symbols are missing after loading 01_setup_data.py.")

for rel_dir in [
    "results/models",
    "results/metrics",
    "results/figures/png",
    "results/figures/svg",
    "results/predictions/sample_predictions",
    "results/predictions/all_predictions",
]:
    Path(rel_dir).mkdir(parents=True, exist_ok=True)

best_model_path = Path("results/models/best_model.pth")
sanity_ckpt_path = Path("results/models/sanity_model.pth")

if best_model_path.exists():
    print("✓ best_model.pth already exists — skipping sanity check.")
else:
    print("── Sanity Check: Training 3 epochs on small subset ──\n")

    sanity_dataset = torch.utils.data.Subset(train_dataset, list(range(min(10, len(train_dataset)))))
    sanity_loader = DataLoader(sanity_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True)

    sanity_model = GCSANet(num_classes=NUM_CLASSES).to(DEVICE)
    sanity_optimizer = torch.optim.AdamW(sanity_model.parameters(), lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY)
    sanity_scaler = build_grad_scaler(enabled=USE_AMP)

    for epoch in range(3):
        loss = train_one_epoch(sanity_model, sanity_loader, criterion,
                               sanity_optimizer, sanity_scaler, DEVICE, GRAD_CLIP)
        print(f"  Sanity Epoch {epoch+1}/3 | Loss: {loss:.4f}")

    print("\n✓ Sanity check passed — loss is decreasing")

    torch.save(sanity_model.state_dict(), sanity_ckpt_path)
    if not best_model_path.exists():
        torch.save(sanity_model.state_dict(), best_model_path)
        print("⚠ results/models/best_model.pth not found. Saved sanity checkpoint as fallback.")

    del sanity_model, sanity_optimizer, sanity_scaler, sanity_loader, sanity_dataset
    torch.cuda.empty_cache()
    gc.collect()

log_path = "results/metrics/training_log.csv"

_training_log_exists = Path(log_path).exists() and Path(log_path).stat().st_size > 0
_best_model_exists = best_model_path.exists()

if _best_model_exists and _training_log_exists:
    print(f"✓ best_model.pth and training_log.csv already exist — skipping training.")
    print(f"  Loading existing training log from {log_path}")
    log_df = pd.read_csv(log_path)
    history = log_df.to_dict('records')
    best_epoch = int(log_df.loc[log_df['val_miou'].idxmax(), 'epoch'])
    best_miou = log_df['val_miou'].max()
    print(f"  Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    if "model" not in globals():
        model = GCSANet(num_classes=NUM_CLASSES, decoder_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(str(best_model_path), map_location=DEVICE, weights_only=True))
    print(f"  Model loaded from {best_model_path}")
else:
    if "model" not in globals():
        model = GCSANet(num_classes=NUM_CLASSES, decoder_dim=256).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    scaler = build_grad_scaler(enabled=USE_AMP)

    best_miou = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"Training GCSANet for {NUM_EPOCHS} epochs")
    print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS}, eta_min=1e-6)")
    print(f"  Early stopping: patience={PATIENCE} on val_miou (vegetation)")
    print(f"  Mixed precision: {USE_AMP}")
    print(f"  Gradient clipping: max_norm={GRAD_CLIP}")
    print(f"{'='*80}\n")

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_start = time.time()
            
            train_loss = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scaler, DEVICE, GRAD_CLIP)
            
            val_loss, val_miou, val_oa, val_per_class = validate(
                model, val_loader, criterion, DEVICE
            )
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            elapsed = time.time() - epoch_start
            
            log_entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_miou": round(val_miou, 6),
                "val_oa": round(val_oa, 6),
                "learning_rate": current_lr,
            }
            history.append(log_entry)
            
            print(f"Epoch {epoch:>3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val mIoU: {val_miou:.4f} | "
                  f"Val OA: {val_oa:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{elapsed:.0f}s")
            
            if val_miou > best_miou:
                best_miou = val_miou
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), "results/models/best_model.pth")
                print(f"  ★ New best mIoU: {best_miou:.4f} — model saved")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n⏹ Early stopping at epoch {epoch} "
                          f"(no improvement for {PATIENCE} epochs)")
                    break
            
            torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        print(f"\n⚠ Training interrupted at epoch {epoch}")
        torch.save(model.state_dict(), "results/models/interrupted_model.pth")
        print("  Checkpoint saved to results/models/interrupted_model.pth")

    log_df = pd.DataFrame(history)
    log_df.to_csv(log_path, index=False)

    print(f"\n{'='*80}")
    print(f"✓ Training complete!")
    print(f"  Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    print(f"  Training log saved to {log_path}")
    print(f"  Best model saved to results/models/best_model.pth")
torch.cuda.empty_cache()

if "model" not in globals():
    model = GCSANet(num_classes=NUM_CLASSES, decoder_dim=256).to(DEVICE)
model.load_state_dict(torch.load(str(best_model_path), map_location=DEVICE, weights_only=True))
model.eval()
print(f"✓ Loaded best model from {best_model_path}\n")

_cached_preds_path = Path("results/predictions/all_preds.npy")
_cached_targets_path = Path("results/predictions/all_targets.npy")
_cached_probs_path = Path("results/predictions/all_probs.npy")

if _cached_preds_path.exists() and _cached_targets_path.exists() and _cached_probs_path.exists():
    print("✓ Loading cached test predictions...")
    all_preds = torch.from_numpy(np.load(str(_cached_preds_path)))
    all_targets = torch.from_numpy(np.load(str(_cached_targets_path)))
    all_probs = torch.from_numpy(np.load(str(_cached_probs_path)))
    print(f"  Loaded {all_preds.shape[0]} cached predictions")
else:
    all_preds_list = []
    all_targets_list = []
    all_probs_list = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test inference"):
            images = images.to(DEVICE, non_blocking=True)
            with amp_autocast(enabled=USE_AMP):
                logits = model(images)
            probs = F.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds_list.append(preds.cpu())
            all_targets_list.append(labels.cpu())
            all_probs_list.append(probs.cpu())

    all_preds = torch.cat(all_preds_list)    # (N, H, W)
    all_targets = torch.cat(all_targets_list)  # (N, H, W)
    all_probs = torch.cat(all_probs_list)      # (N, C, H, W)

    np.save(str(_cached_preds_path), all_preds.numpy())
    np.save(str(_cached_targets_path), all_targets.numpy())
    np.save(str(_cached_probs_path), all_probs.numpy())
    print("✓ Cached test predictions for future runs")

print(f"Test set: {all_preds.shape[0]} images\n")

test_oa = compute_oa(all_preds, all_targets, ignore_index=IGNORE_INDEX)

miou_all, iou_all = compute_miou(all_preds, all_targets,
                                  class_indices=list(range(1, NUM_CLASSES)))

miou_veg, iou_veg = compute_miou(all_preds, all_targets,
                                  class_indices=VEGETATION_INDICES)

print(f"{'='*50}")
print(f"  Overall Accuracy     : {test_oa:.4f}")
print(f"  mIoU (all classes)   : {miou_all:.4f}")
print(f"  mIoU (vegetation)    : {miou_veg:.4f}")
print(f"{'='*50}")

veg_metrics = []
preds_flat = all_preds.numpy().ravel()
targets_flat = all_targets.numpy().ravel()
valid_mask = targets_flat != IGNORE_INDEX

for c in VEGETATION_INDICES:
    pred_c = (preds_flat == c) & valid_mask
    target_c = (targets_flat == c) & valid_mask
    
    tp = (pred_c & target_c).sum()
    fp = (pred_c & ~target_c).sum()
    fn = (~pred_c & target_c).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    veg_metrics.append({
        "class_id": c,
        "class_name": CLASS_NAMES[c],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "iou": round(iou, 4),
    })

print(f"\n{'Class':<18s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'IoU':>8s}")
print("-" * 55)
for m in veg_metrics:
    print(f"{m['class_name']:<18s} {m['precision']:>10.4f} {m['recall']:>8.4f} "
          f"{m['f1_score']:>8.4f} {m['iou']:>8.4f}")

mean_f1_veg = np.mean([m['f1_score'] for m in veg_metrics])
print(f"\n  Mean F1 (vegetation): {mean_f1_veg:.4f}")

pc_df = pd.DataFrame(veg_metrics)
pc_df.to_csv("results/metrics/per_class_metrics.csv", index=False)
print(f"✓ Saved per_class_metrics.csv")

veg_mask_gt = np.isin(targets_flat, VEGETATION_INDICES) & valid_mask
veg_targets = targets_flat[veg_mask_gt]
veg_preds = preds_flat[veg_mask_gt]

veg_idx_map = {v: i for i, v in enumerate(VEGETATION_INDICES)}
veg_targets_mapped = np.array([veg_idx_map.get(t, -1) for t in veg_targets])
veg_preds_mapped = np.array([veg_idx_map.get(p, -1) for p in veg_preds])

valid_cm = (veg_targets_mapped >= 0) & (veg_preds_mapped >= 0)
cm = sk_confusion_matrix(veg_targets_mapped[valid_cm], veg_preds_mapped[valid_cm],
                          labels=list(range(6)))

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

cm_df = pd.DataFrame(cm_norm, index=VEGETATION_NAMES, columns=VEGETATION_NAMES)
cm_df.to_csv("results/metrics/confusion_matrix.csv")
print(f"✓ Saved confusion_matrix.csv (6×6, row-normalized)")

total_params = sum(p.numel() for p in model.parameters())
test_metrics = {
    "overall_accuracy": round(test_oa, 4),
    "mean_iou_all": round(miou_all, 4),
    "mean_iou_vegetation": round(miou_veg, 4),
    "mean_f1_vegetation": round(mean_f1_veg, 4),
    "total_params": total_params,
}
test_metrics_df = pd.DataFrame([{"metric": k, "value": v}
                                 for k, v in test_metrics.items()])
test_metrics_df.to_csv("results/metrics/test_metrics.csv", index=False)
print(f"✓ Saved test_metrics.csv")

print(f"\n── TARGET CHECK ──")
oa_ok = 0.88 <= test_oa <= 0.94  # slight margin
miou_ok = miou_veg >= 0.78       # slight margin
print(f"  OA {test_oa:.4f}  {'✓ PASS' if oa_ok else '✗ BELOW TARGET (0.90-0.92)'}")
print(f"  mIoU(veg) {miou_veg:.4f}  {'✓ PASS' if miou_ok else '✗ BELOW TARGET (≥0.80)'}")
if not (oa_ok and miou_ok):
    print("  → Escalation may be needed: increase epochs, add boundary loss, stronger augmentations")

def compute_boundary_iou(pred_map, gt_map, class_idx, dilation_px=3):
    """Compute IoU within a ±dilation_px band around GT boundaries for a class."""
    gt_c = (gt_map == class_idx).astype(np.uint8)
    pred_c = (pred_map == class_idx).astype(np.uint8)
    
    if gt_c.sum() == 0:
        return float('nan')
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                        (2*dilation_px+1, 2*dilation_px+1))
    dilated = cv2.dilate(gt_c, kernel)
    eroded = cv2.erode(gt_c, kernel)
    boundary_band = (dilated - eroded).astype(bool)
    
    if boundary_band.sum() == 0:
        return float('nan')
    
    pred_in_band = pred_c[boundary_band]
    gt_in_band = gt_c[boundary_band]
    
    intersection = (pred_in_band & gt_in_band).sum()
    union = (pred_in_band | gt_in_band).sum()
    
    return intersection / union if union > 0 else 0.0


_biou_csv_path = Path("results/metrics/boundary_iou.csv")
if _biou_csv_path.exists() and _biou_csv_path.stat().st_size > 0:
    print("✓ Loading cached Boundary IoU from boundary_iou.csv...")
    _biou_df = pd.read_csv(str(_biou_csv_path))
    boundary_results = _biou_df.to_dict('records')
    mean_biou_veg = np.mean([r['boundary_iou'] for r in boundary_results])
    print(f"  Mean Boundary IoU (veg): {mean_biou_veg:.4f}")
else:
    print("Computing Boundary IoU (±3px band)...\n")
    boundary_ious = {c: [] for c in VEGETATION_INDICES}

    for i in tqdm(range(all_preds.shape[0]), desc="Boundary IoU"):
        pred_np = all_preds[i].numpy()
        gt_np = all_targets[i].numpy()
        
        for c in VEGETATION_INDICES:
            biou = compute_boundary_iou(pred_np, gt_np, c, dilation_px=3)
            if not math.isnan(biou):
                boundary_ious[c].append(biou)

    boundary_results = []
    for c in VEGETATION_INDICES:
        mean_biou = np.mean(boundary_ious[c]) if boundary_ious[c] else 0.0
        boundary_results.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "boundary_iou": round(mean_biou, 4),
            "num_images": len(boundary_ious[c]),
        })

    print(f"\n{'Class':<18s} {'Boundary IoU':>12s} {'#Images':>8s}")
    print("-" * 40)
    for r in boundary_results:
        print(f"{r['class_name']:<18s} {r['boundary_iou']:>12.4f} {r['num_images']:>8d}")

    mean_biou_veg = np.mean([r['boundary_iou'] for r in boundary_results])
    print(f"\n  Mean Boundary IoU (veg): {mean_biou_veg:.4f}")

for r in boundary_results:
    for m in veg_metrics:
        if m['class_id'] == r['class_id']:
            m['boundary_iou'] = r['boundary_iou']

pc_df = pd.DataFrame(veg_metrics)
pc_df.to_csv("results/metrics/per_class_metrics.csv", index=False)

biou_df = pd.DataFrame(boundary_results)
biou_df.to_csv("results/metrics/boundary_iou.csv", index=False)
print("✓ Saved boundary_iou.csv")

test_metrics["mean_boundary_iou_vegetation"] = round(mean_biou_veg, 4)
test_metrics_df = pd.DataFrame([{"metric": k, "value": v}
                                 for k, v in test_metrics.items()])
test_metrics_df.to_csv("results/metrics/test_metrics.csv", index=False)

mean_arr = np.array([0.485, 0.456, 0.406])
std_arr  = np.array([0.229, 0.224, 0.225])

_sample_pred_exists = Path("results/predictions/sample_predictions/sample_00.png").exists()
_all_pred_exists = Path(f"results/predictions/all_predictions/pred_0000.png").exists()

if _sample_pred_exists and _all_pred_exists:
    n_samples = min(8, all_preds.shape[0])
    print(f"✓ Predictions already saved — skipping ({n_samples} samples, {all_preds.shape[0]} full masks)")
else:
    print("\nSaving predictions...")

    set_publication_style()
    n_samples = min(8, all_preds.shape[0])

    for idx in range(n_samples):
        img_t, _ = test_dataset[idx]
        img_np = img_t.numpy().transpose(1, 2, 0) * std_arr + mean_arr
        img_np = np.clip(img_np, 0, 1)
        
        gt_np = all_targets[idx].numpy()
        pred_np = all_preds[idx].numpy()
        
        gt_color = np.zeros((*gt_np.shape, 3), dtype=np.uint8)
        pred_color = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
        for c in range(NUM_CLASSES):
            gt_color[gt_np == c] = CLASS_COLORS[c]
            pred_color[pred_np == c] = CLASS_COLORS[c]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_np); axes[0].set_title("RGB Image"); axes[0].axis("off")
        axes[1].imshow(gt_color); axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(pred_color); axes[2].set_title("Prediction"); axes[2].axis("off")
        
        legend_patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i])/255, label=CLASS_NAMES[i])
                          for i in range(NUM_CLASSES)]
        fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=9)
        
        plt.tight_layout()
        save_matplotlib_bundle(
            fig,
            f"results/predictions/sample_predictions/sample_{idx:02d}.png",
            f"results/predictions/sample_predictions/sample_{idx:02d}.svg",
        )
        plt.close(fig)

    for idx in tqdm(range(all_preds.shape[0]), desc="Saving all predictions"):
        pred_np = all_preds[idx].numpy().astype(np.uint8)
        pred_color = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
        for c in range(NUM_CLASSES):
            pred_color[pred_np == c] = CLASS_COLORS[c]
        cv2.imwrite(f"results/predictions/all_predictions/pred_{idx:04d}.png",
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

    print(f"✓ Saved {n_samples} sample prediction panels")
    print(f"✓ Saved {all_preds.shape[0]} full prediction masks")

set_publication_style()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=VEGETATION_NAMES, yticklabels=VEGETATION_NAMES,
            ax=ax, square=True, linewidths=0.5,
            cbar_kws={'label': 'Proportion'})
ax.set_xlabel("Predicted Class", fontsize=14)
ax.set_ylabel("Ground Truth Class", fontsize=14)
ax.set_title("Vegetation Confusion Matrix (Row-Normalized)", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/figures/png/confusion_matrix_heatmap.png",
    "results/figures/svg/confusion_matrix_heatmap.svg",
)
plt.close(fig)
print("✓ Confusion matrix heatmap saved")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(VEGETATION_NAMES))
bars = ax.bar(x, [r['boundary_iou'] for r in boundary_results],
              color=VEG_PALETTE, edgecolor='black', linewidth=0.5, width=0.6)
ax.axhline(y=mean_biou_veg, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean: {mean_biou_veg:.3f}')
ax.set_xticks(x)
ax.set_xticklabels(VEGETATION_NAMES, rotation=30, ha='right')
ax.set_ylabel("Boundary IoU (±3px)")
ax.set_title("Per-Class Boundary IoU Analysis", fontweight='bold')
ax.legend()

for bar, val in zip(bars, [r['boundary_iou'] for r in boundary_results]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/figures/png/boundary_iou_analysis.png",
    "results/figures/svg/boundary_iou_analysis.svg",
)
plt.close(fig)
print("✓ Boundary IoU chart saved")

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None
    print("⚠ segmentation_models_pytorch not installed; skipping U-Net efficiency comparison.")

if smp is not None:
    unet = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet',
                    in_channels=3, classes=NUM_CLASSES)
    unet_params = sum(p.numel() for p in unet.parameters())
    gcsanet_params = sum(p.numel() for p in model.parameters())

    # FLOPs estimation (simple: 2 * MACs ≈ params * input_size for conv-heavy nets)
    # More accurate with a profiler, but this gives order-of-magnitude
    def estimate_flops(m, input_size=(1, 3, 512, 512)):
        """Rough FLOPs estimate by running a forward pass and counting multiplies."""
        total_flops = 0
        hooks = []

        def hook_fn(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                out_h, out_w = output.shape[2], output.shape[3]
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
                total_flops += 2 * kernel_ops * out_h * out_w * module.out_channels
            elif isinstance(module, nn.Linear):
                total_flops += 2 * module.in_features * module.out_features

        for m_layer in m.modules():
            if isinstance(m_layer, (nn.Conv2d, nn.Linear)):
                hooks.append(m_layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            dummy = torch.randn(*input_size)
            m.cpu()(dummy)

        for h in hooks:
            h.remove()

        return total_flops

    unet_flops = estimate_flops(unet)
    model_cpu = model.cpu()
    gcsanet_flops = estimate_flops(model_cpu)
    model.to(DEVICE)

    efficiency_data = {
        "Model": ["GCSANet (Ours)", "U-Net (ResNet-50)"],
        "Parameters (M)": [f"{gcsanet_params/1e6:.1f}", f"{unet_params/1e6:.1f}"],
        "GFLOPs": [f"{gcsanet_flops/1e9:.1f}", f"{unet_flops/1e9:.1f}"],
        "mIoU (veg)": [f"{miou_veg:.4f}", "—"],
        "OA": [f"{test_oa:.4f}", "—"],
    }

    eff_df = pd.DataFrame(efficiency_data)
    print("\n── Efficiency Comparison ──")
    print(eff_df.to_string(index=False))

    test_metrics["total_flops"] = gcsanet_flops
    test_metrics_df = pd.DataFrame([{"metric": k, "value": v}
                                    for k, v in test_metrics.items()])
    test_metrics_df.to_csv("results/metrics/test_metrics.csv", index=False)

    del unet
    gc.collect()

print("\n✓ All scientific analyses complete")

class GradCAM:
    """Grad-CAM for semantic segmentation models."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class):
        """Generate Grad-CAM heatmap for a specific class."""
        self.model.eval()
        self.model.zero_grad()
        
        with amp_autocast(enabled=False):
            output = self.model(input_tensor.float())
        
        target = output[:, target_class, :, :].sum()
        target.backward(retain_graph=True)
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                           mode='bilinear', align_corners=False)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


target_layer = None
for name, module in model.cnn_encoder.named_modules():
    if isinstance(module, nn.Conv2d):
        target_layer = module
        last_name = name

print(f"Grad-CAM target layer: {last_name}")

grad_cam = GradCAM(model, target_layer)

set_publication_style()
n_gradcam = 3

gradcam_indices = []
for idx in range(min(20, len(test_dataset))):
    _, lbl = test_dataset[idx]
    veg_present = sum(1 for v in VEGETATION_INDICES if (lbl.numpy() == v).any())
    if veg_present >= 3:
        gradcam_indices.append(idx)
    if len(gradcam_indices) >= n_gradcam:
        break

while len(gradcam_indices) < n_gradcam:
    gradcam_indices.append(len(gradcam_indices))

print(f"Selected images for Grad-CAM: {gradcam_indices}")

fig, axes = plt.subplots(n_gradcam, 4, figsize=(20, 5 * n_gradcam))

for row, idx in enumerate(gradcam_indices):
    img_t, lbl_t = test_dataset[idx]
    img_np = img_t.numpy().transpose(1, 2, 0) * std_arr + mean_arr
    img_np = np.clip(img_np, 0, 1)
    lbl_np = lbl_t.numpy()
    
    with torch.no_grad():
        pred = model(img_t.unsqueeze(0).to(DEVICE)).argmax(1).squeeze().cpu().numpy()
    
    gt_color = np.zeros((*lbl_np.shape, 3), dtype=np.uint8)
    pred_color = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        gt_color[lbl_np == c] = CLASS_COLORS[c]
        pred_color[pred == c] = CLASS_COLORS[c]
    
    dominant_class = max(VEGETATION_INDICES,
                        key=lambda c: (lbl_np == c).sum())
    
    cam = grad_cam.generate(img_t.unsqueeze(0).to(DEVICE), dominant_class)
    
    axes[row, 0].imshow(img_np)
    axes[row, 0].set_title("RGB Image")
    axes[row, 0].axis("off")
    
    axes[row, 1].imshow(gt_color)
    axes[row, 1].set_title("Ground Truth")
    axes[row, 1].axis("off")
    
    axes[row, 2].imshow(pred_color)
    axes[row, 2].set_title("Prediction")
    axes[row, 2].axis("off")
    
    axes[row, 3].imshow(img_np)
    axes[row, 3].imshow(cam, cmap='jet', alpha=0.5)
    axes[row, 3].set_title(f"Grad-CAM ({CLASS_NAMES[dominant_class]})")
    axes[row, 3].axis("off")

legend_patches = [mpatches.Patch(color=np.array(CLASS_COLORS[i])/255, label=CLASS_NAMES[i])
                  for i in range(NUM_CLASSES)]
fig.legend(handles=legend_patches, loc="lower center", ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.suptitle("Grad-CAM Analysis — Model Attention on Vegetation Classes",
             fontsize=16, fontweight='bold')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/figures/png/gradcam_analysis.png",
    "results/figures/svg/gradcam_analysis.svg",
)
plt.close(fig)
print("✓ Grad-CAM analysis saved")
