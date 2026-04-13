"""Model definitions, losses, and training utilities for GCSANet."""

import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

DEVICE = globals().get("DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
NUM_CLASSES = globals().get("NUM_CLASSES", 10)
BATCH_SIZE = globals().get("BATCH_SIZE", 8)
USE_AMP = globals().get("USE_AMP", torch.cuda.is_available())
AMP_DEVICE_TYPE = globals().get("AMP_DEVICE_TYPE", "cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = globals().get("IGNORE_INDEX", 0)
LABEL_SMOOTHING = globals().get("LABEL_SMOOTHING", 0.1)
VEGETATION_INDICES = globals().get("VEGETATION_INDICES", [3, 5, 6, 7, 8, 9])
CLASS_WEIGHTS = globals().get(
    "CLASS_WEIGHTS",
    torch.ones(NUM_CLASSES, dtype=torch.float32, device=DEVICE),
)


def amp_autocast(enabled):
    return torch.amp.autocast(device_type=AMP_DEVICE_TYPE, enabled=enabled)

class GatedCrossScaleAttention(nn.Module):
    """Gated fusion of CNN and Transformer features at a single scale.
    
    Learns a sigmoid gate per channel that dynamically weights
    local (CNN) vs global (Transformer) information.
    
        F_fused = G * F_cnn + (1 - G) * F_trans
        where G = sigmoid(Conv(Cat(F_cnn, F_trans)))
        """
    
    def __init__(self, cnn_channels, trans_channels, out_channels):
        super().__init__()
        self.align_cnn = nn.Conv2d(cnn_channels, out_channels, 1, bias=False) \
            if cnn_channels != out_channels else nn.Identity()
        self.align_trans = nn.Conv2d(trans_channels, out_channels, 1, bias=False) \
            if trans_channels != out_channels else nn.Identity()
        
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, f_cnn, f_trans):
        f_cnn = self.align_cnn(f_cnn)
        f_trans = self.align_trans(f_trans)
        
        if f_cnn.shape[2:] != f_trans.shape[2:]:
            f_trans = F.interpolate(f_trans, size=f_cnn.shape[2:],
                                    mode='bilinear', align_corners=False)
        
        cat = torch.cat([f_cnn, f_trans], dim=1)
        gate = self.gate(cat)
        fused = gate * f_cnn + (1 - gate) * f_trans
        
        return self.refine(fused)


class FPNBlock(nn.Module):
    """Single FPN decoder block: upsample + lateral add + refine."""
    
    def __init__(self, in_channels, lateral_channels, out_channels):
        super().__init__()
        self.lateral = nn.Conv2d(lateral_channels, out_channels, 1, bias=False)
        self.upsample_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, lateral):
        x = self.upsample_conv(x)
        x = F.interpolate(x, size=lateral.shape[2:], mode='bilinear', align_corners=False)
        lateral = self.lateral(lateral)
        return self.refine(x + lateral)


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder with 4-scale inputs."""
    
    def __init__(self, encoder_channels, decoder_dim=256):
        """
        Args:
            encoder_channels: list of 4 ints, channels at each scale [s1, s2, s3, s4]
                              from highest to lowest resolution.
            decoder_dim: unified channel dim in decoder.
        """
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        
        self.top = nn.Conv2d(c4, decoder_dim, 1, bias=False)
        self.fpn3 = FPNBlock(decoder_dim, c3, decoder_dim)
        self.fpn2 = FPNBlock(decoder_dim, c2, decoder_dim)
        self.fpn1 = FPNBlock(decoder_dim, c1, decoder_dim)
        
        self.merge = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
    
    def forward(self, features):
        """features: list [f1, f2, f3, f4] from high-res to low-res."""
        f1, f2, f3, f4 = features
        
        p4 = self.top(f4)
        p3 = self.fpn3(p4, f3)
        p2 = self.fpn2(p3, f2)
        p1 = self.fpn1(p2, f1)
        
        target_size = p1.shape[2:]
        p2_up = F.interpolate(p2, size=target_size, mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        
        merged = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
        return self.merge(merged)


class GCSANet(nn.Module):
    """Gated Cross-Scale Attention Network.
    
    Dual-encoder (EfficientNet-B4 + MiT-B2) with per-scale gated fusion
    and FPN decoder for 10-class LULC semantic segmentation.
    """
    
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
        
        print(f"  CNN encoder channels:  {cnn_channels}")
        print(f"  Trans encoder channels: {trans_channels}")
        
        fusion_channels = [max(c, t) for c, t in zip(cnn_channels, trans_channels)]
        
        self.gcsa_modules = nn.ModuleList([
            GatedCrossScaleAttention(cc, tc, fc)
            for cc, tc, fc in zip(cnn_channels, trans_channels, fusion_channels)
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
        input_size = x.shape[2:]
        
        cnn_feats = self.cnn_encoder(x)
        trans_feats = self.trans_encoder(x)
        
        fused = [
            gcsa(f_cnn, f_trans)
            for gcsa, f_cnn, f_trans in zip(self.gcsa_modules, cnn_feats, trans_feats)
        ]
        
        decoded = self.decoder(fused)
        
        logits = self.seg_head(decoded)
        logits = F.interpolate(logits, size=input_size,
                               mode='bilinear', align_corners=False)
        
        return logits


print("✓ GCSANet architecture defined")

model = GCSANet(num_classes=NUM_CLASSES, decoder_dim=256).to(DEVICE)

with torch.no_grad():
    dummy = torch.randn(BATCH_SIZE, 3, 512, 512, device=DEVICE)
    if USE_AMP:
        with amp_autocast(enabled=True):
            output = model(dummy)
    else:
        output = model(dummy)

print(f"✓ Forward pass successful")
print(f"  Input shape  : {dummy.shape}")
print(f"  Output shape : {output.shape}  (expect [{BATCH_SIZE}, {NUM_CLASSES}, 512, 512])")
assert output.shape == (BATCH_SIZE, NUM_CLASSES, 512, 512), "Output shape mismatch!"

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Total parameters    : {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size          : {total_params * 4 / 1e6:.1f} MB (FP32)")

del dummy, output
torch.cuda.empty_cache()

class FocalLoss(nn.Module):
    """Focal Loss with ignore_index, class weights, and label smoothing.
    
    Properly masks ignored pixels BEFORE applying label smoothing.
    """
    
    def __init__(self, gamma=2.0, weight=None, ignore_index=0,
                 label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) int64
        """
        B, C, H, W = logits.shape
        
        valid_mask = (targets != self.ignore_index)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        valid_flat = valid_mask.reshape(-1)
        
        logits_valid = logits_flat[valid_flat]
        targets_valid = targets_flat[valid_flat]
        
        num_classes = C
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits_valid)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets_valid.unsqueeze(1),
                                    1.0 - self.label_smoothing)
        
        log_probs = F.log_softmax(logits_valid, dim=1)
        probs = torch.exp(log_probs)
        
        true_probs = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - true_probs) ** self.gamma
        
        loss_per_pixel = -(smooth_targets * log_probs).sum(dim=1)
        loss_per_pixel = focal_weight * loss_per_pixel
        
        if self.weight is not None:
            class_w = self.weight[targets_valid]
            loss_per_pixel = loss_per_pixel * class_w
        
        return loss_per_pixel.mean()


class DiceLoss(nn.Module):
    """Soft Dice Loss with explicit ignore_index masking.
    
    Computes per-class Dice and averages over non-ignored classes.
    """
    
    def __init__(self, ignore_index=0, smooth=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) int64
        """
        B, C, H, W = logits.shape
        
        valid_mask = (targets != self.ignore_index).float()
        
        probs = F.softmax(logits, dim=1)
        
        targets_onehot = F.one_hot(targets.clamp(0, C - 1), C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        
        valid_mask_4d = valid_mask.unsqueeze(1)
        probs = probs * valid_mask_4d
        targets_onehot = targets_onehot * valid_mask_4d
        
        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_onehot.sum(dim=dims)
        
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        
        dice_per_class = dice_per_class[1:]
        
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """Focal Loss + Dice Loss combined."""
    
    def __init__(self, weight=None, ignore_index=0, label_smoothing=0.1,
                 focal_gamma=2.0, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, weight=weight,
                               ignore_index=ignore_index,
                               label_smoothing=label_smoothing)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, logits, targets):
        fl = self.focal(logits, targets)
        dl = self.dice(logits, targets)
        return self.focal_weight * fl + self.dice_weight * dl


criterion = CombinedLoss(
    weight=CLASS_WEIGHTS,
    ignore_index=IGNORE_INDEX,
    label_smoothing=LABEL_SMOOTHING,
    focal_gamma=2.0,
).to(DEVICE)

print("✓ Combined Loss (Focal + Dice) instantiated")
print(f"  Focal: gamma=2.0, label_smoothing=0.1, class weights applied")
print(f"  Dice: explicit ignore_index=0 masking")
print(f"  Both components exclude Invalid (class 0)")

def compute_miou(preds, targets, num_classes=NUM_CLASSES,
                 class_indices=None, ignore_index=0):
    """Compute mean IoU for specified classes, ignoring index 0.
    
    Args:
        preds: (B, H, W) int64 predictions
        targets: (B, H, W) int64 ground truth
        class_indices: list of class indices to compute mIoU over.
                       If None, use all except ignore_index.
    Returns:
        miou: float
        per_class_iou: dict {class_idx: iou}
    """
    if class_indices is None:
        class_indices = list(range(1, num_classes))
    
    per_class_iou = {}
    for c in class_indices:
        pred_c = (preds == c)
        target_c = (targets == c)
        valid = (targets != ignore_index)
        
        pred_c = pred_c & valid
        target_c = target_c & valid
        
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        
        if union > 0:
            per_class_iou[c] = (intersection / union).item()
        else:
            per_class_iou[c] = float('nan')
    
    valid_ious = [v for v in per_class_iou.values() if not math.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    return miou, per_class_iou


def compute_oa(preds, targets, ignore_index=0):
    """Overall accuracy excluding ignored pixels."""
    valid = (targets != ignore_index)
    if valid.sum() == 0:
        return 0.0
    correct = ((preds == targets) & valid).sum().float()
    return (correct / valid.sum().float()).item()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip=1.0):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with amp_autocast(enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    return epoch_loss


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and compute loss, mIoU (veg), OA."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with amp_autocast(enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)
        
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())
    
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    miou, per_class = compute_miou(all_preds, all_targets,
                                    class_indices=VEGETATION_INDICES)
    oa = compute_oa(all_preds, all_targets)
    
    return epoch_loss, miou, oa, per_class


print("✓ Training utilities defined (train_one_epoch, validate, compute_miou, compute_oa)")
