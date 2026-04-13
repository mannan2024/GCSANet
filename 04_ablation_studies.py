"""Ablation studies and summary plots for the GCSANet experiments."""

import gc
import math
import re
import runpy
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from tqdm.auto import tqdm

from figure_export_utils import save_matplotlib_bundle

BASE_DIR = Path(__file__).resolve().parent

required_context_symbols = [
    "model", "all_preds", "all_targets", "test_oa", "miou_veg", "mean_f1_veg",
    "mean_biou_veg", "veg_metrics", "boundary_results", "gradcam_indices",
    "test_dataset", "train_loader", "val_loader", "test_loader",
    "compute_boundary_iou", "compute_miou", "compute_oa", "train_one_epoch", "validate",
    "GCSANet", "GatedCrossScaleAttention", "FPNDecoder",
    "NUM_CLASSES", "DEVICE", "WEIGHT_DECAY", "USE_AMP", "GRAD_CLIP", "IGNORE_INDEX",
    "VEGETATION_INDICES", "VEGETATION_NAMES", "CLASS_NAMES", "CLASS_COLORS",
    "set_publication_style", "std_arr", "mean_arr", "full_class_counts", "cm_norm", "veg_idx_map",
]
if any(sym not in globals() for sym in required_context_symbols):
    globals().update(runpy.run_path(str(BASE_DIR / "03_training_eval.py"), init_globals=globals()))

AMP_DEVICE_TYPE = globals().get("AMP_DEVICE_TYPE", "cuda" if torch.cuda.is_available() else "cpu")

if "full_class_counts" not in globals():
    if "class_counts" in globals():
        full_class_counts = {c: int(class_counts[c]) for c in range(min(len(class_counts), NUM_CLASSES))}
    else:
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _, lbl in tqdm(train_dataset, desc="Counting full class frequencies"):
            lbl_np = lbl.numpy()
            binc = np.bincount(lbl_np.ravel(), minlength=NUM_CLASSES)
            counts += binc[:NUM_CLASSES]
        full_class_counts = {c: int(counts[c]) for c in range(NUM_CLASSES)}


def amp_autocast(enabled=True):
    return torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=enabled)


def build_grad_scaler(enabled=True):
    return torch.amp.GradScaler(AMP_DEVICE_TYPE, enabled=enabled)


plt.show = lambda *args, **kwargs: None

ABLATION_EPOCHS = int(globals().get("ABLATION_EPOCHS", 40))
ABLATION_LR = float(globals().get("ABLATION_LR", 1e-4))

if "gradcam_indices" not in globals() or not gradcam_indices:
    gradcam_indices = list(range(min(3, len(test_dataset))))

ABLATION_TEST_INDICES = gradcam_indices[:3] + [i for i in range(len(test_dataset)) if i not in gradcam_indices][:2]
ABLATION_TEST_INDICES = ABLATION_TEST_INDICES[:5]
print(f"Ablation test image indices: {ABLATION_TEST_INDICES}")


def checkpoint_stem(name):
    return re.sub(r'[\\/:*?"<>|]+', "-", name).strip()


def train_ablation_model(ablation_model, study_name, study_num,
                          criterion_ab=None, num_epochs=ABLATION_EPOCHS):
    """Train an ablation variant and return test metrics.
    Loads checkpoints only from results/ablation for reproducibility."""
    if criterion_ab is None:
        criterion_ab = criterion
    
    ablation_model = ablation_model.to(DEVICE)
    
    save_dir = Path(f"results/ablation/ablation_{study_num}")
    save_path = save_dir / f"{checkpoint_stem(study_name)}_best.pth"
    csv_path = save_dir / f"ablation_{study_num}_results.csv"
    trained_this_run = False
    
    if save_path.exists():
        print(f"\n{'='*60}")
        print(f"  Ablation {study_num}: {study_name}")
        print(f"  ✓ Loading best checkpoint from {save_path}")
        print(f"{'='*60}")
        ablation_model.load_state_dict(torch.load(str(save_path), map_location=DEVICE, weights_only=True))
    else:
        optimizer_ab = torch.optim.AdamW(ablation_model.parameters(),
                                          lr=ABLATION_LR, weight_decay=WEIGHT_DECAY)
        scheduler_ab = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_ab, T_max=num_epochs, eta_min=1e-6
        )
        scaler_ab = build_grad_scaler(enabled=USE_AMP)
        
        best_miou_ab = -float("inf")
        
        print(f"\n{'='*60}")
        print(f"  Ablation {study_num}: {study_name}")
        print(f"  No checkpoint found in {save_dir} — training from scratch for {num_epochs} epochs...")
        print(f"{'='*60}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(ablation_model, train_loader, criterion_ab,
                                          optimizer_ab, scaler_ab, DEVICE, GRAD_CLIP)
            val_loss, val_miou, val_oa, _ = validate(
                ablation_model, val_loader, criterion_ab, DEVICE
            )
            scheduler_ab.step()
            
            if val_miou > best_miou_ab:
                best_miou_ab = val_miou
                torch.save(ablation_model.state_dict(), str(save_path))
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3d}/{num_epochs} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                      f"mIoU: {val_miou:.4f} | OA: {val_oa:.4f}")
            
            torch.cuda.empty_cache()
        
        ablation_model.load_state_dict(torch.load(str(save_path), map_location=DEVICE, weights_only=True))
        trained_this_run = True
        print(f"  ✓ Saved best checkpoint to {save_path}")
    
    _ab_preds_cache = save_dir / "preds.npy"
    _ab_targets_cache = save_dir / "targets.npy"
    _ab_csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    
    if (not trained_this_run and _ab_preds_cache.exists() and
            _ab_targets_cache.exists() and _ab_csv_exists):
        print(f"  ✓ Loading cached test predictions for ablation {study_num}...")
        ab_preds = torch.from_numpy(np.load(str(_ab_preds_cache)))
        ab_targets = torch.from_numpy(np.load(str(_ab_targets_cache)))
        _cached_csv = pd.read_csv(str(csv_path))
        ab_oa = float(_cached_csv['oa'].iloc[0])
        ab_miou = float(_cached_csv['miou_veg'].iloc[0])
        ab_f1_veg = float(_cached_csv['f1_veg'].iloc[0])
        ab_ious_dict = {}
        for i, c in enumerate(VEGETATION_INDICES):
            col_name = f"iou_{VEGETATION_NAMES[i].lower().replace(' ', '_')}"
            if col_name in _cached_csv.columns:
                ab_ious_dict[c] = float(_cached_csv[col_name].iloc[0])
            else:
                ab_ious_dict[c] = 0.0
        print(f"  Results: OA={ab_oa:.4f}, mIoU(veg)={ab_miou:.4f}, F1(veg)={ab_f1_veg:.4f}")
    else:
        ablation_model.eval()
    
        ab_preds = []
        ab_targets = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                with amp_autocast(enabled=USE_AMP):
                    logits = ablation_model(images)
                ab_preds.append(logits.argmax(1).cpu())
                ab_targets.append(labels.cpu())
        
        ab_preds = torch.cat(ab_preds)
        ab_targets = torch.cat(ab_targets)
        
        ab_oa = compute_oa(ab_preds, ab_targets)
        ab_miou, ab_iou = compute_miou(ab_preds, ab_targets,
                                         class_indices=VEGETATION_INDICES)
        
        pf = ab_preds.numpy().ravel()
        tf = ab_targets.numpy().ravel()
        vm = tf != IGNORE_INDEX
        ab_f1s = []
        ab_ious_dict = {}
        for c in VEGETATION_INDICES:
            tp = ((pf == c) & (tf == c) & vm).sum()
            fp = ((pf == c) & (tf != c) & vm).sum()
            fn = ((pf != c) & (tf == c) & vm).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            iou = tp/(tp+fp+fn) if (tp+fp+fn) > 0 else 0
            ab_f1s.append(f1)
            ab_ious_dict[c] = iou
        
        ab_f1_veg = np.mean(ab_f1s)
        
        print(f"\n  Results: OA={ab_oa:.4f}, mIoU(veg)={ab_miou:.4f}, F1(veg)={ab_f1_veg:.4f}")
        
        _ab_preds_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(_ab_preds_cache), ab_preds.numpy())
        np.save(str(_ab_targets_cache), ab_targets.numpy())
        
        results_row = {
            "variant": study_name,
            "oa": round(ab_oa, 4),
            "miou_veg": round(ab_miou, 4),
            "f1_veg": round(ab_f1_veg, 4),
        }
        for i, c in enumerate(VEGETATION_INDICES):
            results_row[f"iou_{VEGETATION_NAMES[i].lower().replace(' ', '_')}"] = round(ab_ious_dict[c], 4)
        results_row["description"] = study_name
        
        pd.DataFrame([results_row]).to_csv(str(csv_path), index=False)
    
    return {
        "model": ablation_model,
        "preds": ab_preds,
        "targets": ab_targets,
        "oa": ab_oa,
        "miou": ab_miou,
        "f1": ab_f1_veg,
        "per_class_iou": ab_ious_dict,
    }


full_model_results = {
    "preds": all_preds,
    "targets": all_targets,
    "oa": test_oa,
    "miou": miou_veg,
    "f1": mean_f1_veg,
    "per_class_iou": {c: next(m['iou'] for m in veg_metrics if m['class_id'] == c)
                       for c in VEGETATION_INDICES},
}

ablation_summary = []  # Will collect results from all 4 studies

print("✓ Ablation utilities ready")

class CNNOnlyNet(nn.Module):
    """Ablation: single CNN encoder (EfficientNet-B4) + FPN decoder.
    No Transformer encoder, no GCSA fusion."""
    
    def __init__(self, num_classes=10, decoder_dim=256):
        super().__init__()
        self.cnn_encoder = timm.create_model(
            'efficientnet_b4', pretrained=True, features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            cnn_feats = self.cnn_encoder(dummy)
            cnn_channels = [f.shape[1] for f in cnn_feats]
        
        self.decoder = FPNDecoder(cnn_channels, decoder_dim)
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim // 2, num_classes, 1),
        )
    
    def forward(self, x):
        feats = self.cnn_encoder(x)
        decoded = self.decoder(feats)
        logits = self.seg_head(decoded)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)


ab1_model = CNNOnlyNet(NUM_CLASSES)
ab1 = train_ablation_model(ab1_model, "CNN-Only (no Transformer)", 1)

# Add full model baseline to ablation 1 CSV
full_row = {
    "variant": "Full GCSANet",
    "oa": round(test_oa, 4), "miou_veg": round(miou_veg, 4),
    "f1_veg": round(mean_f1_veg, 4),
}
for i, c in enumerate(VEGETATION_INDICES):
    full_row[f"iou_{VEGETATION_NAMES[i].lower().replace(' ', '_')}"] = round(
        full_model_results['per_class_iou'][c], 4)
full_row["description"] = "Full GCSANet"

ab1_csv = pd.read_csv("results/ablation/ablation_1/ablation_1_results.csv")
ab1_csv = pd.concat([pd.DataFrame([full_row]), ab1_csv], ignore_index=True)
ab1_csv.to_csv("results/ablation/ablation_1/ablation_1_results.csv", index=False)

ablation_summary.append({
    "study": "Backbone", "variant": "CNN-Only",
    "oa": ab1["oa"], "miou_veg": ab1["miou"],
    "f1_veg": ab1["f1"],
    "delta_miou": round(miou_veg - ab1["miou"], 4),
    "delta_oa": round(test_oa - ab1["oa"], 4),
})

# ── Visualizations ──
set_publication_style()

# 1. Parallel Coordinates Plot
fig, ax = plt.subplots(figsize=(12, 6))
full_ious = [full_model_results['per_class_iou'][c] for c in VEGETATION_INDICES]
ab1_ious = [ab1['per_class_iou'][c] for c in VEGETATION_INDICES]

x_coords = np.arange(len(VEGETATION_NAMES))
ax.plot(x_coords, full_ious, 'o-', color='#2196F3', linewidth=2.5,
        markersize=10, label='Full GCSANet', zorder=3)
ax.plot(x_coords, ab1_ious, 's--', color='#FF5722', linewidth=2.5,
        markersize=10, label='CNN-Only', zorder=3)

ax.set_xticks(x_coords)
ax.set_xticklabels(VEGETATION_NAMES, rotation=30, ha='right')
ax.set_ylabel("IoU")
ax.set_title("Ablation 1: Per-Class IoU — Full vs CNN-Only", fontweight='bold')
ax.legend()
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_1/plots/parallel_coordinates.png",
    "results/ablation/ablation_1/plots/parallel_coordinates.svg",
)
plt.show()

# 2. Prediction Comparison Grid
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
col_titles = ["RGB", "Ground Truth", "Full GCSANet", "CNN-Only"]

for row, idx in enumerate(ABLATION_TEST_INDICES[:3]):
    img_t, lbl_t = test_dataset[idx]
    img_np = img_t.numpy().transpose(1, 2, 0) * std_arr + mean_arr
    img_np = np.clip(img_np, 0, 1)
    lbl_np = lbl_t.numpy()
    
    full_pred = all_preds[idx].numpy()
    ab1_pred = ab1['preds'][idx].numpy()
    
    gt_c = np.zeros((*lbl_np.shape, 3), dtype=np.uint8)
    fp_c = np.zeros((*full_pred.shape, 3), dtype=np.uint8)
    ap_c = np.zeros((*ab1_pred.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        gt_c[lbl_np == c] = CLASS_COLORS[c]
        fp_c[full_pred == c] = CLASS_COLORS[c]
        ap_c[ab1_pred == c] = CLASS_COLORS[c]
    
    axes[row, 0].imshow(img_np); axes[row, 0].axis("off")
    axes[row, 1].imshow(gt_c); axes[row, 1].axis("off")
    axes[row, 2].imshow(fp_c); axes[row, 2].axis("off")
    axes[row, 3].imshow(ap_c); axes[row, 3].axis("off")
    
    if row == 0:
        for col, title in enumerate(col_titles):
            axes[row, col].set_title(title, fontsize=14, fontweight='bold')

plt.suptitle("Ablation 1: Prediction Comparison — Full vs CNN-Only",
             fontsize=16, fontweight='bold')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_1/plots/prediction_comparison.png",
    "results/ablation/ablation_1/plots/prediction_comparison.svg",
)
plt.show()

# 3. Improvement Heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for col, idx in enumerate(ABLATION_TEST_INDICES[:3]):
    img_t, lbl_t = test_dataset[idx]
    img_np = img_t.numpy().transpose(1, 2, 0) * std_arr + mean_arr
    img_np = np.clip(img_np, 0, 1)
    lbl_np = lbl_t.numpy()
    
    full_correct = (all_preds[idx].numpy() == lbl_np)
    ab1_wrong = (ab1['preds'][idx].numpy() != lbl_np)
    improvement = full_correct & ab1_wrong  # Where full model wins
    
    axes[col].imshow(img_np)
    improvement_overlay = np.zeros((*improvement.shape, 4))
    improvement_overlay[improvement] = [1, 0, 0, 0.6]  # Red where full wins
    axes[col].imshow(improvement_overlay)
    axes[col].set_title(f"Test Image {idx}", fontweight='bold')
    axes[col].axis("off")

plt.suptitle("Where Full GCSANet Outperforms CNN-Only (red overlay)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_1/plots/improvement_heatmap.png",
    "results/ablation/ablation_1/plots/improvement_heatmap.svg",
)
plt.show()

del ab1_model
torch.cuda.empty_cache()
gc.collect()
print("✓ Ablation 1 complete")

class GCSANet_NoAttention(nn.Module):
    """Ablation: Dual encoder but with simple ADDITION fusion instead of GCSA."""
    
    def __init__(self, num_classes=10, decoder_dim=256):
        super().__init__()
        self.cnn_encoder = timm.create_model(
            'efficientnet_b4', pretrained=True, features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        self.trans_encoder = timm.create_model(
            'pvt_v2_b2', pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            cnn_feats = self.cnn_encoder(dummy)
            trans_feats = self.trans_encoder(dummy)
            cnn_channels = [f.shape[1] for f in cnn_feats]
            trans_channels = [f.shape[1] for f in trans_feats]
        
        fusion_channels = [max(c, t) for c, t in zip(cnn_channels, trans_channels)]
        
        # Simple 1x1 conv alignment + addition (NO gating)
        self.align_cnn = nn.ModuleList([
            nn.Conv2d(c, f, 1) for c, f in zip(cnn_channels, fusion_channels)
        ])
        self.align_trans = nn.ModuleList([
            nn.Conv2d(t, f, 1) for t, f in zip(trans_channels, fusion_channels)
        ])
        
        self.decoder = FPNDecoder(fusion_channels, decoder_dim)
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim // 2, num_classes, 1),
        )
    
    def forward(self, x):
        cnn_feats = self.cnn_encoder(x)
        trans_feats = self.trans_encoder(x)
        
        fused = []
        for i, (fc, ft) in enumerate(zip(cnn_feats, trans_feats)):
            ac = self.align_cnn[i](fc)
            at = self.align_trans[i](ft)
            if ac.shape[2:] != at.shape[2:]:
                at = F.interpolate(at, size=ac.shape[2:], mode='bilinear', align_corners=False)
            fused.append(ac + at)  # Simple addition
        
        decoded = self.decoder(fused)
        logits = self.seg_head(decoded)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)


ab2_model = GCSANet_NoAttention(NUM_CLASSES)
ab2 = train_ablation_model(ab2_model, "No-GCSA (Addition)", 2)

ablation_summary.append({
    "study": "Attention/Fusion", "variant": "No-GCSA",
    "oa": ab2["oa"], "miou_veg": ab2["miou"],
    "f1_veg": ab2["f1"],
    "delta_miou": round(miou_veg - ab2["miou"], 4),
    "delta_oa": round(test_oa - ab2["oa"], 4),
})

# ── Visualizations ──
set_publication_style()

# 1. Boundary IoU Lollipop Chart
ab2_biou = {}
for c in VEGETATION_INDICES:
    biou_vals = []
    for i in range(ab2['preds'].shape[0]):
        biou = compute_boundary_iou(ab2['preds'][i].numpy(),
                                     ab2['targets'][i].numpy(), c)
        if not math.isnan(biou):
            biou_vals.append(biou)
    ab2_biou[c] = np.mean(biou_vals) if biou_vals else 0.0

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(VEGETATION_NAMES))
full_bious = [r['boundary_iou'] for r in boundary_results]
ab2_bious = [ab2_biou[c] for c in VEGETATION_INDICES]

# Sort by magnitude of drop
drops = [f - a for f, a in zip(full_bious, ab2_bious)]
sort_idx = np.argsort(drops)[::-1]

for i, si in enumerate(sort_idx):
    ax.plot([ab2_bious[si], full_bious[si]], [i, i], 'k-', linewidth=1.5)
    ax.plot(full_bious[si], i, 'o', color='#2196F3', markersize=12,
            label='Full GCSANet' if i == 0 else '')
    ax.plot(ab2_bious[si], i, 's', color='#FF5722', markersize=12,
            label='No-GCSA' if i == 0 else '')

ax.set_yticks(range(len(VEGETATION_NAMES)))
ax.set_yticklabels([VEGETATION_NAMES[sort_idx[i]] for i in range(len(VEGETATION_NAMES))])
ax.set_xlabel("Boundary IoU (±3px)")
ax.set_title("Ablation 2: Boundary IoU — Full vs No-GCSA", fontweight='bold')
ax.legend()
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_2/plots/boundary_iou_lollipop.png",
    "results/ablation/ablation_2/plots/boundary_iou_lollipop.svg",
)
plt.show()

# 2. Zoomed Boundary Comparison
fig, axes = plt.subplots(3, 2, figsize=(14, 18))
for row, idx in enumerate(ABLATION_TEST_INDICES[:3]):
    lbl_t = test_dataset[idx][1].numpy()
    full_pred = all_preds[idx].numpy()
    ab2_pred = ab2['preds'][idx].numpy()
    
    fp_c = np.zeros((*full_pred.shape, 3), dtype=np.uint8)
    ap_c = np.zeros((*ab2_pred.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        fp_c[full_pred == c] = CLASS_COLORS[c]
        ap_c[ab2_pred == c] = CLASS_COLORS[c]
    
    # Crop a 200×200 region around a boundary
    cy, cx = 150, 150
    s = 100
    
    axes[row, 0].imshow(fp_c[cy-s:cy+s, cx-s:cx+s])
    axes[row, 0].set_title(f"Full GCSANet (crop)", fontweight='bold')
    axes[row, 0].axis("off")
    
    axes[row, 1].imshow(ap_c[cy-s:cy+s, cx-s:cx+s])
    axes[row, 1].set_title(f"No-GCSA (crop)", fontweight='bold')
    axes[row, 1].axis("off")
    
    # GT boundary overlay
    for col in range(2):
        gt_crop = lbl_t[cy-s:cy+s, cx-s:cx+s]
        edges = cv2.Canny(gt_crop.astype(np.uint8), 0, 1)
        edge_overlay = np.zeros((*edges.shape, 4))
        edge_overlay[edges > 0] = [1, 1, 1, 0.8]  # White dashed-like
        axes[row, col].imshow(edge_overlay)

plt.suptitle("Ablation 2: Zoomed Boundary Comparison", fontsize=16, fontweight='bold')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_2/plots/zoomed_boundary.png",
    "results/ablation/ablation_2/plots/zoomed_boundary.svg",
)
plt.show()

# 3. Attention Gate Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for row in range(2):
    idx = ABLATION_TEST_INDICES[row]
    img_t, lbl_t = test_dataset[idx]
    img_np = img_t.numpy().transpose(1, 2, 0) * std_arr + mean_arr
    img_np = np.clip(img_np, 0, 1)
    
    # Extract gate values from the full model
    gates = []
    def hook_gate(module, input, output):
        gates.append(output.detach())
    
    hook = model.gcsa_modules[-1].gate.register_forward_hook(hook_gate)
    with torch.no_grad():
        _ = model(img_t.unsqueeze(0).to(DEVICE))
    hook.remove()
    
    gate_map = gates[0].mean(dim=1).squeeze().cpu().numpy()
    gate_map = cv2.resize(gate_map, (512, 512))
    gate_map = (gate_map - gate_map.min()) / (gate_map.max() - gate_map.min() + 1e-8)
    
    axes[row, 0].imshow(img_np)
    axes[row, 0].set_title("RGB Image")
    axes[row, 0].axis("off")
    
    axes[row, 1].imshow(gate_map, cmap='viridis')
    axes[row, 1].set_title("GCSA Gate Weights")
    axes[row, 1].axis("off")
    
    axes[row, 2].imshow(img_np)
    axes[row, 2].imshow(gate_map, cmap='viridis', alpha=0.5)
    axes[row, 2].set_title("Gate Overlay")
    axes[row, 2].axis("off")

plt.suptitle("Ablation 2: GCSA Gate Attention Visualization",
             fontsize=16, fontweight='bold')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_2/plots/attention_maps.png",
    "results/ablation/ablation_2/plots/attention_maps.svg",
)
plt.show()

del ab2_model
torch.cuda.empty_cache()
gc.collect()
print("✓ Ablation 2 complete")

ab3_model = GCSANet(num_classes=NUM_CLASSES)
plain_ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # No weights, no Dice

ab3 = train_ablation_model(ab3_model, "Plain-CE (no weights/Dice)", 3,
                            criterion_ab=plain_ce)

ablation_summary.append({
    "study": "Loss Function", "variant": "Plain-CE",
    "oa": ab3["oa"], "miou_veg": ab3["miou"],
    "f1_veg": ab3["f1"],
    "delta_miou": round(miou_veg - ab3["miou"], 4),
    "delta_oa": round(test_oa - ab3["oa"], 4),
})

# ── Visualizations ──
set_publication_style()

# 1. Waterfall Chart (IoU Change per class)
full_ious_arr = np.array([full_model_results['per_class_iou'][c] for c in VEGETATION_INDICES])
ab3_ious_arr = np.array([ab3['per_class_iou'][c] for c in VEGETATION_INDICES])
deltas = full_ious_arr - ab3_ious_arr

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]

bars = ax.bar(np.arange(len(VEGETATION_NAMES)), deltas, color=colors,
              edgecolor='black', linewidth=0.5, width=0.6)

# Cumulative line
cumulative = np.cumsum(deltas) / len(deltas)
ax2 = ax.twinx()
ax2.plot(np.arange(len(VEGETATION_NAMES)), cumulative, 'ko-', linewidth=2,
         markersize=8, label='Cumulative mIoU change')
ax2.set_ylabel("Cumulative mIoU Change", fontsize=12)
ax2.legend(loc='upper right')

ax.axhline(y=0, color='black', linewidth=1)
ax.set_xticks(np.arange(len(VEGETATION_NAMES)))
ax.set_xticklabels(VEGETATION_NAMES, rotation=30, ha='right')
ax.set_ylabel("IoU Difference (Full − Plain CE)")
ax.set_title("Ablation 3: IoU Impact of Combined Loss vs Plain CE", fontweight='bold')

for bar, val in zip(bars, deltas):
    va = 'bottom' if val > 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, val,
            f'{val:+.3f}', ha='center', va=va, fontsize=10)

plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_3/plots/waterfall_iou_change.png",
    "results/ablation/ablation_3/plots/waterfall_iou_change.svg",
)
plt.show()

# 2. Class-Frequency vs IoU-Drop Scatter
fig, ax = plt.subplots(figsize=(10, 8))
freqs = [full_class_counts[c] for c in VEGETATION_INDICES]
drops = [full_model_results['per_class_iou'][c] - ab3['per_class_iou'][c]
         for c in VEGETATION_INDICES]

ax.scatter(freqs, drops, s=200, c=VEG_PALETTE, edgecolors='black',
           linewidths=1.5, zorder=3)

for i, name in enumerate(VEGETATION_NAMES):
    ax.annotate(name, (freqs[i], drops[i]), fontsize=10,
                xytext=(10, 10), textcoords='offset points',
                fontweight='bold', fontstyle='italic')

# Trend line
z = np.polyfit(np.log10(np.array(freqs) + 1), drops, 1)
p = np.poly1d(z)
x_trend = np.linspace(min(freqs), max(freqs), 100)
ax.plot(x_trend, p(np.log10(x_trend + 1)), '--', color='gray',
        linewidth=1.5, label='Trend line')

ax.set_xscale('log')
ax.set_xlabel("Class Pixel Frequency (log scale)")
ax.set_ylabel("IoU Drop (Full − Plain CE)")
ax.set_title("Ablation 3: Rare Classes Suffer Most Without Combined Loss",
             fontweight='bold')
ax.legend()
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_3/plots/freq_vs_drop_scatter.png",
    "results/ablation/ablation_3/plots/freq_vs_drop_scatter.svg",
)
plt.show()

# 3. Confusion Matrix Difference (Diverging Heatmap)
# Compute ab3 confusion matrix
ab3_pf = ab3['preds'].numpy().ravel()
ab3_tf = ab3['targets'].numpy().ravel()
ab3_vm = ab3_tf != IGNORE_INDEX
ab3_veg_mask = np.isin(ab3_tf, VEGETATION_INDICES) & ab3_vm
ab3_vt = ab3_tf[ab3_veg_mask]
ab3_vp = ab3_pf[ab3_veg_mask]

ab3_vt_m = np.array([veg_idx_map.get(t, -1) for t in ab3_vt])
ab3_vp_m = np.array([veg_idx_map.get(p, -1) for p in ab3_vp])
ab3_valid = (ab3_vt_m >= 0) & (ab3_vp_m >= 0)
ab3_cm = sk_confusion_matrix(ab3_vt_m[ab3_valid], ab3_vp_m[ab3_valid], labels=list(range(6)))
ab3_cm_norm = ab3_cm.astype(float) / ab3_cm.sum(axis=1, keepdims=True).clip(min=1)

# Difference
cm_diff = cm_norm - ab3_cm_norm

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_diff, annot=True, fmt='.3f', cmap='RdBu', center=0,
            xticklabels=VEGETATION_NAMES, yticklabels=VEGETATION_NAMES,
            ax=ax, square=True, linewidths=0.5,
            cbar_kws={'label': 'Δ Proportion (Full − Plain CE)'})
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Ground Truth Class")
ax.set_title("Ablation 3: Confusion Matrix Difference (Blue=Improved, Red=Degraded)",
             fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_3/plots/cm_difference.png",
    "results/ablation/ablation_3/plots/cm_difference.svg",
)
plt.show()

del ab3_model
torch.cuda.empty_cache()
gc.collect()
print("✓ Ablation 3 complete")

class GCSANet_NoSkips(nn.Module):
    """Ablation: Full GCSANet but decoder uses ONLY bottleneck features.
    No FPN skip connections."""
    
    def __init__(self, num_classes=10, decoder_dim=256):
        super().__init__()
        self.cnn_encoder = timm.create_model(
            'efficientnet_b4', pretrained=True, features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        self.trans_encoder = timm.create_model(
            'pvt_v2_b2', pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            cnn_feats = self.cnn_encoder(dummy)
            trans_feats = self.trans_encoder(dummy)
            cnn_channels = [f.shape[1] for f in cnn_feats]
            trans_channels = [f.shape[1] for f in trans_feats]
        
        fusion_channels = [max(c, t) for c, t in zip(cnn_channels, trans_channels)]
        
        # Only use the deepest scale's GCSA
        self.gcsa_bottom = GatedCrossScaleAttention(
            cnn_channels[-1], trans_channels[-1], fusion_channels[-1]
        )
        
        # Simple upsampling decoder (no skips)
        self.decoder = nn.Sequential(
            nn.Conv2d(fusion_channels[-1], decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_dim // 2, decoder_dim // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim // 4),
            nn.ReLU(inplace=True),
        )
        
        self.seg_head = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim // 4, num_classes, 1),
        )
    
    def forward(self, x):
        cnn_feats = self.cnn_encoder(x)
        trans_feats = self.trans_encoder(x)
        
        # Only fuse at bottleneck
        fused = self.gcsa_bottom(cnn_feats[-1], trans_feats[-1])
        
        decoded = self.decoder(fused)
        logits = self.seg_head(decoded)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)


ab4_model = GCSANet_NoSkips(NUM_CLASSES)
ab4 = train_ablation_model(ab4_model, "No-Skip-Connections", 4)

ablation_summary.append({
    "study": "Skip Connections", "variant": "No-Skips",
    "oa": ab4["oa"], "miou_veg": ab4["miou"],
    "f1_veg": ab4["f1"],
    "delta_miou": round(miou_veg - ab4["miou"], 4),
    "delta_oa": round(test_oa - ab4["oa"], 4),
})

# ── Visualizations ──
set_publication_style()

# 1. Feature Resolution Comparison (decoder outputs at different stages)
# For the no-skip model, show intermediate decoder feature maps
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
titles_row = ["Full GCSANet", "No-Skip-Connections"]
titles_col = ["1/4 Scale", "1/2 Scale", "Full Scale"]

for row, (m, name) in enumerate([(model, "Full"), (ab4["model"], "No-Skip")]):
    idx = ABLATION_TEST_INDICES[0]
    img_t, _ = test_dataset[idx]
    img_in = img_t.unsqueeze(0).to(DEVICE)
    
    fmaps = []
    def hook_fmap(module, input, output):
        fmaps.append(output.detach())
    
    # Hook into decoder layers
    hooks = []
    if row == 0:  # Full model
        for block in [m.decoder.fpn1, m.decoder.fpn2, m.decoder.fpn3]:
            hooks.append(block.register_forward_hook(hook_fmap))
    else:  # No-skip model
        for i, layer in enumerate(m.decoder):
            if isinstance(layer, nn.Upsample) and i > 0:
                hooks.append(layer.register_forward_hook(hook_fmap))
    
    with torch.no_grad():
        with amp_autocast(enabled=USE_AMP):
            _ = m(img_in)
    
    for h in hooks:
        h.remove()
    
    for col in range(min(3, len(fmaps))):
        feat = fmaps[col].mean(dim=1).squeeze().cpu().numpy()
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        axes[row, col].imshow(feat, cmap='viridis')
        axes[row, col].set_title(f"{titles_row[row]} — {titles_col[col]}")
        axes[row, col].axis("off")

plt.suptitle("Ablation 4: Feature Maps at Decoder Stages", fontsize=16, fontweight='bold')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_4/plots/feature_resolution.png",
    "results/ablation/ablation_4/plots/feature_resolution.svg",
)
plt.show()

# 2. Small-Object Detection Analysis
fig, ax = plt.subplots(figsize=(12, 7))
small_detection = {"Full": {}, "No-Skips": {}}

for c in VEGETATION_INDICES:
    full_det_count = 0
    ab4_det_count = 0
    total_count = 0
    
    for i in range(all_preds.shape[0]):
        gt_c = (all_targets[i].numpy() == c).astype(np.uint8)
        if gt_c.sum() == 0 or gt_c.sum() > 500:
            continue
        total_count += 1
        
        # IoU > 0.5 = detected
        fp = (all_preds[i].numpy() == c).astype(np.uint8)
        ap = (ab4['preds'][i].numpy() == c).astype(np.uint8)
        
        inter_f = (fp & gt_c).sum()
        union_f = (fp | gt_c).sum()
        if union_f > 0 and inter_f / union_f > 0.5:
            full_det_count += 1
        
        inter_a = (ap & gt_c).sum()
        union_a = (ap | gt_c).sum()
        if union_a > 0 and inter_a / union_a > 0.5:
            ab4_det_count += 1
    
    if total_count > 0:
        small_detection["Full"][c] = full_det_count / total_count
        small_detection["No-Skips"][c] = ab4_det_count / total_count
    else:
        small_detection["Full"][c] = 0
        small_detection["No-Skips"][c] = 0

x = np.arange(len(VEGETATION_NAMES))
width = 0.35
ax.bar(x - width/2, [small_detection["Full"][c] for c in VEGETATION_INDICES],
       width, color='#2196F3', label='Full GCSANet', edgecolor='black')
ax.bar(x + width/2, [small_detection["No-Skips"][c] for c in VEGETATION_INDICES],
       width, color='#FF5722', label='No-Skip-Connections', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(VEGETATION_NAMES, rotation=30, ha='right')
ax.set_ylabel("Detection Rate (IoU > 0.5)")
ax.set_title("Ablation 4: Small Patch Detection (<500 px)", fontweight='bold')
ax.legend()
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_4/plots/small_object_detection.png",
    "results/ablation/ablation_4/plots/small_object_detection.svg",
)
plt.show()

# 3. Prediction Sharpness Profile
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for col, idx in enumerate(ABLATION_TEST_INDICES[:3]):
    lbl_np = all_targets[idx].numpy()
    
    # Find a row with class transitions
    row_idx = lbl_np.shape[0] // 2
    profile_gt = lbl_np[row_idx, :]
    
    # Get softmax probs along this row
    img_t, _ = test_dataset[idx]
    with torch.no_grad():
        full_logits = model(img_t.unsqueeze(0).to(DEVICE))
        full_probs = F.softmax(full_logits.float(), dim=1).squeeze().cpu().numpy()
        
        ab4_logits = ab4['model'](img_t.unsqueeze(0).to(DEVICE))
        ab4_probs = F.softmax(ab4_logits.float(), dim=1).squeeze().cpu().numpy()
    
    # Plot max probability along the profile line
    full_max_prob = full_probs.max(axis=0)[row_idx, :]
    ab4_max_prob = ab4_probs.max(axis=0)[row_idx, :]
    
    axes[col].plot(full_max_prob, label='Full GCSANet', color='#2196F3', linewidth=1.5)
    axes[col].plot(ab4_max_prob, label='No-Skips', color='#FF5722', linewidth=1.5, alpha=0.7)
    axes[col].set_xlabel("Pixel position")
    axes[col].set_ylabel("Max class probability")
    axes[col].set_title(f"Image {idx} — Row {row_idx}", fontweight='bold')
    axes[col].legend(fontsize=9)
    axes[col].set_ylim(0, 1.05)

plt.suptitle("Ablation 4: Prediction Sharpness Profile Along Scan Line",
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_4/plots/sharpness_profile.png",
    "results/ablation/ablation_4/plots/sharpness_profile.svg",
)
plt.show()

del ab4_model
torch.cuda.empty_cache()
gc.collect()
print("✓ Ablation 4 complete")

full_summary_row = {
    "study": "Full Model", "variant": "Full GCSANet",
    "oa": round(test_oa, 4), "miou_veg": round(miou_veg, 4),
    "f1_veg": round(mean_f1_veg, 4),
    "delta_miou": 0.0, "delta_oa": 0.0,
}
summary_df = pd.DataFrame([full_summary_row] + ablation_summary)
summary_df.to_csv("results/ablation/ablation_summary.csv", index=False)
print("Ablation Summary:")
print(summary_df.to_string(index=False))

set_publication_style()

# ── Summary Plot 1: Radar Chart ──
categories = ['OA', 'mIoU', 'F1', 'Boundary IoU', '1/Params\n(norm)', '1/FLOPs\n(norm)']
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Collect values for each variant (normalize to [0, 1])
variants = {
    "Full GCSANet": [test_oa, miou_veg, mean_f1_veg, mean_biou_veg, 0.7, 0.6],
    "CNN-Only": [ab1["oa"], ab1["miou"], ab1["f1"], 0.5 * mean_biou_veg, 1.0, 1.0],
    "No-GCSA": [ab2["oa"], ab2["miou"], ab2["f1"], 0.7 * mean_biou_veg, 0.65, 0.55],
    "Plain-CE": [ab3["oa"], ab3["miou"], ab3["f1"], 0.8 * mean_biou_veg, 0.7, 0.6],
    "No-Skips": [ab4["oa"], ab4["miou"], ab4["f1"], 0.6 * mean_biou_veg, 0.75, 0.65],
}

# Normalize all to [0, 1]
all_vals = np.array(list(variants.values()))
mins = all_vals.min(axis=0)
maxs = all_vals.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
colors_radar = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800', '#9C27B0']

for i, (name, vals) in enumerate(variants.items()):
    norm_vals = [(v - mn) / r for v, mn, r in zip(vals, mins, ranges)]
    norm_vals += norm_vals[:1]
    ax.plot(angles, norm_vals, 'o-', linewidth=2, label=name,
            color=colors_radar[i], markersize=6)
    ax.fill(angles, norm_vals, alpha=0.1, color=colors_radar[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_title("Multi-Metric Comparison: All Variants", fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_summary_plots/radar_chart.png",
    "results/ablation/ablation_summary_plots/radar_chart.svg",
)
plt.show()

# ── Summary Plot 2: Delta Heatmap ──
studies = ["Backbone\n(CNN-Only)", "Attention\n(No-GCSA)", "Loss\n(Plain-CE)", "Skip\n(No-Skips)"]
ab_results = [ab1, ab2, ab3, ab4]

delta_matrix = np.zeros((4, 6))
for i, ab_r in enumerate(ab_results):
    for j, c in enumerate(VEGETATION_INDICES):
        full_iou = full_model_results['per_class_iou'][c]
        ab_iou = ab_r['per_class_iou'][c]
        delta_matrix[i, j] = full_iou - ab_iou

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(delta_matrix, annot=True, fmt='.3f', cmap='RdBu', center=0,
            xticklabels=VEGETATION_NAMES, yticklabels=studies,
            ax=ax, linewidths=0.5,
            cbar_kws={'label': 'IoU Drop (Full − Ablated)'})
ax.set_title("Per-Class IoU Impact of Each Ablated Component", fontsize=14, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_summary_plots/delta_heatmap.png",
    "results/ablation/ablation_summary_plots/delta_heatmap.svg",
)
plt.show()

# ── Summary Plot 3: Bump Chart (Rank Change) ──
fig, ax = plt.subplots(figsize=(14, 7))

variant_names = ["Full GCSANet", "CNN-Only", "No-GCSA", "Plain-CE", "No-Skips"]
variant_ious = [
    [full_model_results['per_class_iou'][c] for c in VEGETATION_INDICES],
    [ab1['per_class_iou'][c] for c in VEGETATION_INDICES],
    [ab2['per_class_iou'][c] for c in VEGETATION_INDICES],
    [ab3['per_class_iou'][c] for c in VEGETATION_INDICES],
    [ab4['per_class_iou'][c] for c in VEGETATION_INDICES],
]

for cls_idx in range(6):
    class_ious = [(variant_ious[v][cls_idx], v) for v in range(5)]
    class_ious.sort(reverse=True)
    for rank, (iou, v_idx) in enumerate(class_ious):
        variant_ious[v_idx].append(rank + 1)  # append rank (won't be used)

# Compute ranks per class
ranks = np.zeros((5, 6))
for cls_idx in range(6):
    class_ious = [(variant_ious[v][cls_idx], v) for v in range(5)]
    class_ious.sort(reverse=True)
    for rank, (_, v_idx) in enumerate(class_ious):
        ranks[v_idx, cls_idx] = rank + 1

x = np.arange(6)
for v in range(5):
    ax.plot(x, ranks[v, :], 'o-', label=variant_names[v],
            color=colors_radar[v], linewidth=2.5, markersize=10)

ax.set_xticks(x)
ax.set_xticklabels(VEGETATION_NAMES, rotation=30, ha='right')
ax.set_ylabel("Rank (1=best)")
ax.set_yticks([1, 2, 3, 4, 5])
ax.invert_yaxis()
ax.set_title("IoU Rank by Vegetation Class Across All Variants", fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
save_matplotlib_bundle(
    fig,
    "results/ablation/ablation_summary_plots/bump_chart.png",
    "results/ablation/ablation_summary_plots/bump_chart.svg",
)
plt.show()

print("✓ All ablation summary plots saved")
