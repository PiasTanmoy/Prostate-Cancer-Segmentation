# ...existing imports...
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import cv2
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn as nn
import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torch
import SimpleITK as sitk
import nibabel as nib
import numpy as np

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from torch.utils.data import Dataset
import torch

import torch.nn.functional as F

# --- Constants ---
NUM_CLASSES = 2  # 0,1,2,3,4,5 (but only 0,2,3,4,5 used)
CLASS_LABELS = [0, 1]  # 1 is unused, but needed for indexing
print(f"Number of classes: {NUM_CLASSES}")

TARGET_H, TARGET_W = 256, 256   # wanted height & width
print(f"Target size: {TARGET_H}x{TARGET_W}")


# --- Dataset ---
class LazyProstateSegDataset(Dataset):
    def __init__(self, img_root, mask_root, cases_df, transform=None):
        self.transform = transform
        self.entries = []
        for _, row in cases_df.iterrows():
            patient_id = str(row["patient_id"])
            study_id = str(row["study_id"])
            patient_path = f"{img_root}/{patient_id}/{patient_id}_{study_id}_t2w.mha"
            mask_path = f"{mask_root}/{patient_id}_{study_id}.nii.gz"
            if not (os.path.exists(patient_path) and os.path.exists(mask_path)):
                continue
            # For each slice in the volume
            img_volume = sitk.GetArrayFromImage(sitk.ReadImage(patient_path))
            mask_volume = nib.load(mask_path).get_fdata()
            mask_volume = np.transpose(mask_volume, (2, 0, 1)) if mask_volume.shape != img_volume.shape else mask_volume
            for z in range(img_volume.shape[0]):
                self.entries.append({
                    'img_path': patient_path,
                    'mask_path': mask_path,
                    'slice_idx': z
                })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_volume = sitk.GetArrayFromImage(sitk.ReadImage(entry['img_path']))
        mask_volume = nib.load(entry['mask_path']).get_fdata()
        mask_volume = np.transpose(mask_volume, (2, 0, 1)) if mask_volume.shape != img_volume.shape else mask_volume
        z = entry['slice_idx']
        img_slice = img_volume[z].astype(np.float32)
        mask_slice = mask_volume[z].astype(np.int64)

        # Map 2,3,4,5 to 1, 0 stays 0
        mask_slice = np.where(np.isin(mask_slice, [2,3,4,5]), 1, 0).astype(np.int64)

        img_slice = (img_slice - np.mean(img_slice)) / (np.std(img_slice) + 1e-6)
        img_slice = cv2.resize(img_slice, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        mask_slice = cv2.resize(mask_slice, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        # Add channel dimension for image
        img_slice = np.expand_dims(img_slice, axis=0)
        img_tensor = torch.tensor(img_slice, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.long)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, mask_tensor

# --- Model ---
class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.out(d2)  # logits

# --- Class Weights for Imbalance ---
def compute_class_weights(dataset):
    counts = np.zeros(NUM_CLASSES)
    for _, mask in DataLoader(dataset, batch_size=1):
        unique, cnts = torch.unique(mask, return_counts=True)
        for u, c in zip(unique, cnts):
            if u < NUM_CLASSES:
                counts[u] += c.item()
    ## Set unused class (1) count to a large value so its weight is near zero
    counts[1] = 1e12
    # Avoid division by zero
    weights = 1.0 / (counts + 1e-6)
    # Only normalize over used classes (0,2,3,4,5)
    used = [0,2,3,4,5]
    weights[used] = weights[used] / weights[used].sum() * len(used)
    weights[1] = 0.0  # Explicitly ignore class 1
    return torch.tensor(weights, dtype=torch.float32).to(device)

def compute_class_weights_from_df():
    # Read the CSV
    freq_df = pd.read_csv("mask_pixel_frequencies.csv")
    counts = freq_df.iloc[0].values.astype(np.float64)  # [0,2,3,4,5]

    # If you want weights for NUM_CLASSES=6 (0-5, with 1 unused)
    NUM_CLASSES = 6
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    # Assign counts to correct indices (0,2,3,4,5)
    weights[[0,2,3,4,5]] = counts

    # Compute class weights: inverse frequency, normalized over used classes
    used = [0,2,3,4,5]
    inv = 1.0 / (weights[used] + 1e-6)
    normed = inv / inv.sum() * len(used)
    final_weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    final_weights[used] = normed
    final_weights[1] = 0.0  # Unused class

    print("Class weights:", final_weights)
    class_weights_tensor = torch.tensor(final_weights, dtype=torch.float32)
    
    return class_weights_tensor

def compute_binary_class_weights_from_df():
    freq_df = pd.read_csv("mask_pixel_frequencies.csv")
    counts = freq_df.iloc[0].values.astype(np.float64)  # [0,2,3,4,5]
    bg_count = counts[0]
    cancer_count = counts[1:].sum()
    weights = np.array([bg_count, cancer_count], dtype=np.float64)
    inv = 1.0 / (weights + 1e-6)
    normed = inv / inv.sum() * 2
    class_weights_tensor = torch.tensor(normed, dtype=torch.float32)
    print("Binary class weights:", class_weights_tensor)
    return class_weights_tensor


# --- Loss ---
def get_loss_fn(class_weights):
    return nn.CrossEntropyLoss(weight=class_weights)  # ignore unused class 1

# --- Metrics ---
def multiclass_dice(pred, target, num_classes=NUM_CLASSES, eps=1e-6):
    pred = torch.argmax(pred, dim=1)  # [B,H,W]
    dice_scores = []
    for cls in [2,3,4,5]:  # only cancer classes
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        inter = (pred_cls * target_cls).sum(dim=(1,2))
        union = pred_cls.sum(dim=(1,2)) + target_cls.sum(dim=(1,2))
        dice = (2*inter + eps) / (union + eps)
        dice_scores.append(dice.mean().item())
    return np.mean(dice_scores)

def per_class_dice(pred, target, classes=[0,2,3,4,5], eps=1e-6):
    """
    pred: logits [B, C, H, W]
    target: [B, H, W]
    Returns: dict {class: dice}
    """
    pred = torch.argmax(pred, dim=1)  # [B,H,W]
    dices = {}
    unique_classes = torch.unique(target)
    present_classes = [cls for cls in classes if (unique_classes == cls).any()]
    for cls in present_classes:
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        inter = (pred_cls * target_cls).sum(dim=(1,2))
        union = pred_cls.sum(dim=(1,2)) + target_cls.sum(dim=(1,2))
        dice = (2*inter + eps) / (union + eps)
        dices[cls] = dice.mean().item()
    return dices

def multiclass_dice_present(pred, target, classes=[0,1], eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    dices = {cls: [] for cls in classes}
    mean_dices = []
    for b in range(pred.shape[0]):
        sample_dices = []
        unique_classes = torch.unique(target[b])
        present_classes = [cls for cls in classes if (unique_classes == cls).any()]
        for cls in present_classes:
            pred_cls = (pred[b] == cls).float()
            target_cls = (target[b] == cls).float()
            inter = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            dice = (2*inter + eps) / (union + eps)
            dices[cls].append(dice.item())
            sample_dices.append(dice.item())
        mean_dices.append(np.mean(sample_dices) if sample_dices else 0.0)
    avg_dices = {cls: np.mean(dices[cls]) if dices[cls] else None for cls in classes}
    mean_dice = np.mean(mean_dices)
    return mean_dice, avg_dices

def multiclass_iou_present(pred, target, classes=[0,1], eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    ious = {cls: [] for cls in classes}
    mean_ious = []
    for b in range(pred.shape[0]):
        sample_ious = []
        unique_classes = torch.unique(target[b])
        present_classes = [cls for cls in classes if (unique_classes == cls).any()]
        for cls in present_classes:
            pred_cls = (pred[b] == cls).float()
            target_cls = (target[b] == cls).float()
            inter = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - inter
            iou = (inter + eps) / (union + eps)
            ious[cls].append(iou.item())
            sample_ious.append(iou.item())
        mean_ious.append(np.mean(sample_ious) if sample_ious else 0.0)
    avg_ious = {cls: np.mean(ious[cls]) if ious[cls] else None for cls in classes}
    mean_iou = np.mean(mean_ious)
    return mean_iou, avg_ious

# --- Training ---
def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs):
    history = []
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_dice, train_iou = [], [], []
        train_dice_per_class = {cls: [] for cls in [0,2,3,4,5]}
        train_iou_per_class = {cls: [] for cls in [0,2,3,4,5]}
        for batch_idx, (img, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, mask)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            with torch.no_grad():
                mean_dice, per_class_d = multiclass_dice_present(output, mask)
                mean_iou, per_class_i = multiclass_iou_present(output, mask)
                train_dice.append(mean_dice)
                train_iou.append(mean_iou)
                for cls in train_dice_per_class:
                    if cls in per_class_d:
                        train_dice_per_class[cls].append(per_class_d[cls])
                for cls in train_iou_per_class:
                    if cls in per_class_i:
                        train_iou_per_class[cls].append(per_class_i[cls])
            print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")

        print(f"Epoch {epoch+1} | Avg Train Loss: {np.mean(train_loss):.4f} | Avg Train Dice (present): {np.mean(train_dice):.4f} | Avg Train IoU (present): {np.mean(train_iou):.4f}")
        avg_train_dice_per_class = {cls: np.mean(train_dice_per_class[cls]) if train_dice_per_class[cls] else None for cls in train_dice_per_class}
        avg_train_iou_per_class = {cls: np.mean(train_iou_per_class[cls]) if train_iou_per_class[cls] else None for cls in train_iou_per_class}
        print(f"Epoch {epoch+1} | Train Per-Class Dice (present): {avg_train_dice_per_class}")
        print(f"Epoch {epoch+1} | Train Per-Class IoU (present): {avg_train_iou_per_class}")

        # Validation
        model.eval()
        val_loss, val_dice, val_iou = [], [], []
        val_dice_per_class = {cls: [] for cls in [0,2,3,4,5]}
        val_iou_per_class = {cls: [] for cls in [0,2,3,4,5]}
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                output = model(img)
                loss = loss_fn(output, mask)
                val_loss.append(loss.item())
                mean_dice, per_class_d = multiclass_dice_present(output, mask)
                mean_iou, per_class_i = multiclass_iou_present(output, mask)
                val_dice.append(mean_dice)
                val_iou.append(mean_iou)
                for cls in val_dice_per_class:
                    if cls in per_class_d:
                        val_dice_per_class[cls].append(per_class_d[cls])
                for cls in val_iou_per_class:
                    if cls in per_class_i:
                        val_iou_per_class[cls].append(per_class_i[cls])
        avg_val_dice_per_class = {cls: np.mean(val_dice_per_class[cls]) if val_dice_per_class[cls] else None for cls in val_dice_per_class}
        avg_val_iou_per_class = {cls: np.mean(val_iou_per_class[cls]) if val_iou_per_class[cls] else None for cls in val_iou_per_class}
        print(f"[Epoch {epoch+1}] Train Loss: {np.mean(train_loss):.4f}, Val Dice (present): {np.mean(val_dice):.4f}, Val IoU (present): {np.mean(val_iou):.4f}")
        print(f"[Epoch {epoch+1}] Val Per-Class Dice (present): {avg_val_dice_per_class}")
        print(f"[Epoch {epoch+1}] Val Per-Class IoU (present): {avg_val_iou_per_class}")

        # --- Evaluate on test set ---
        avg_test_dice, avg_test_dice_per_class, avg_test_iou, avg_test_iou_per_class, test_case_metrics = evaluate_loader(model, test_loader, loss_fn, split_name="test")
        print(f"[Epoch {epoch+1}] Test Dice (mean over present cancer classes): {avg_test_dice:.4f}")
        print(f"[Epoch {epoch+1}] Test Per-Class Dice (present): {avg_test_dice_per_class}")
        print(f"[Epoch {epoch+1}] Test IoU (mean over present cancer classes): {avg_test_iou:.4f}")
        print(f"[Epoch {epoch+1}] Test Per-Class IoU (present): {avg_test_iou_per_class}")

        # --- Save per-case metrics for train, val, test ---
        _, _, _, _, train_case_metrics = evaluate_loader(model, train_loader, loss_fn, split_name="train")
        _, _, _, _, val_case_metrics = evaluate_loader(model, val_loader, loss_fn, split_name="val")
        pd.DataFrame(train_case_metrics).to_csv(os.path.join(save_dir, f"train_case_metrics_epoch{epoch+1}.csv"), index=False)
        pd.DataFrame(val_case_metrics).to_csv(os.path.join(save_dir, f"val_case_metrics_epoch{epoch+1}.csv"), index=False)
        pd.DataFrame(test_case_metrics).to_csv(os.path.join(save_dir, f"test_case_metrics_epoch{epoch+1}.csv"), index=False)

        history.append({
            "epoch": epoch+1,
            "train_loss": np.mean(train_loss),
            "val_loss": np.mean(val_loss),
            "train_dice": np.mean(train_dice),
            "val_dice": np.mean(val_dice),
            "train_iou": np.mean(train_iou),
            "val_iou": np.mean(val_iou),
            "test_dice": avg_test_dice,
            "test_iou": avg_test_iou,
            **{f"train_dice_cls_{cls}": avg_train_dice_per_class[cls] for cls in avg_train_dice_per_class},
            **{f"val_dice_cls_{cls}": avg_val_dice_per_class[cls] for cls in avg_val_dice_per_class},
            **{f"test_dice_cls_{cls}": avg_test_dice_per_class[cls] for cls in avg_test_dice_per_class},
            **{f"train_iou_cls_{cls}": avg_train_iou_per_class[cls] for cls in avg_train_iou_per_class},
            **{f"val_iou_cls_{cls}": avg_val_iou_per_class[cls] for cls in avg_val_iou_per_class},
            **{f"test_iou_cls_{cls}": avg_test_iou_per_class[cls] for cls in avg_test_iou_per_class},
        })
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch+1}.pth"))
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

# --- Testing ---
def evaluate_test(model, test_loader, loss_fn):
    model.eval()
    test_dice, test_iou = [], []
    test_dice_per_class = {cls: [] for cls in [0,2,3,4,5]}
    test_iou_per_class = {cls: [] for cls in [0,2,3,4,5]}
    with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            mean_dice, per_class_d = multiclass_dice_present(output, mask)
            mean_iou, per_class_i = multiclass_iou_present(output, mask)
            test_dice.append(mean_dice)
            test_iou.append(mean_iou)
            for cls in test_dice_per_class:
                if cls in per_class_d:
                    test_dice_per_class[cls].append(per_class_d[cls])
            for cls in test_iou_per_class:
                if cls in per_class_i:
                    test_iou_per_class[cls].append(per_class_i[cls])
    avg_test_dice_per_class = {cls: np.mean(test_dice_per_class[cls]) if test_dice_per_class[cls] else None for cls in test_dice_per_class}
    avg_test_iou_per_class = {cls: np.mean(test_iou_per_class[cls]) if test_iou_per_class[cls] else None for cls in test_iou_per_class}
    print(f"\n📊 Test Dice (mean over present cancer classes): {np.mean(test_dice):.4f}")
    print(f"📊 Test Per-Class Dice (present): {avg_test_dice_per_class}")
    print(f"📊 Test IoU (mean over present cancer classes): {np.mean(test_iou):.4f}")
    print(f"📊 Test Per-Class IoU (present): {avg_test_iou_per_class}")

def evaluate_loader(model, loader, loss_fn, split_name="test"):
    model.eval()
    all_case_metrics = []
    mean_dice_list, mean_iou_list = [], []
    dice_per_class_dict = {cls: [] for cls in [0,2,3,4,5]}
    iou_per_class_dict = {cls: [] for cls in [0,2,3,4,5]}
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            mean_dice, per_class_dice = multiclass_dice_present(output, mask)
            mean_iou, per_class_iou = multiclass_iou_present(output, mask)
            mean_dice_list.append(mean_dice)
            mean_iou_list.append(mean_iou)
            for cls in dice_per_class_dict:
                if cls in per_class_dice:
                    dice_per_class_dict[cls].append(per_class_dice[cls])
            for cls in iou_per_class_dict:
                if cls in per_class_iou:
                    iou_per_class_dict[cls].append(per_class_iou[cls])
            # Save per-case metrics (for each batch, which may be >1 slices)
            for b in range(img.shape[0]):
                gt_classes = torch.unique(mask[b]).cpu().numpy().tolist()
                row = {
                    "split": split_name,
                    "mean_dice": mean_dice,
                    "mean_iou": mean_iou,
                    "gt_classes": gt_classes
                }
                for cls in [0,2,3,4,5]:
                    row[f"dice_cls_{cls}"] = per_class_dice.get(cls, None)
                    row[f"iou_cls_{cls}"] = per_class_iou.get(cls, None)
                all_case_metrics.append(row)
    avg_dice_per_class = {cls: np.mean(dice_per_class_dict[cls]) if dice_per_class_dict[cls] else None for cls in dice_per_class_dict}
    avg_iou_per_class = {cls: np.mean(iou_per_class_dict[cls]) if iou_per_class_dict[cls] else None for cls in iou_per_class_dict}
    avg_mean_dice = np.mean(mean_dice_list)
    avg_mean_iou = np.mean(mean_iou_list)
    return avg_mean_dice, avg_dice_per_class, avg_mean_iou, avg_iou_per_class, all_case_metrics

# --- Training ---
def train_model(model, train_loader, val_loader, test_loader, optimizer, loss_fn, num_epochs):
    history = []
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_dice, train_iou = [], [], []
        train_dice_per_class = {cls: [] for cls in [0,2,3,4,5]}
        train_iou_per_class = {cls: [] for cls in [0,2,3,4,5]}
        for batch_idx, (img, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, mask)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            with torch.no_grad():
                mean_dice, per_class_d = multiclass_dice_present(output, mask)
                mean_iou, per_class_i = multiclass_iou_present(output, mask)
                train_dice.append(mean_dice)
                train_iou.append(mean_iou)
                for cls in train_dice_per_class:
                    if cls in per_class_d:
                        train_dice_per_class[cls].append(per_class_d[cls])
                for cls in train_iou_per_class:
                    if cls in per_class_i:
                        train_iou_per_class[cls].append(per_class_i[cls])
            print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")

        print(f"Epoch {epoch+1} | Avg Train Loss: {np.mean(train_loss):.4f} | Avg Train Dice (present): {np.mean(train_dice):.4f} | Avg Train IoU (present): {np.mean(train_iou):.4f}")
        avg_train_dice_per_class = {cls: np.mean(train_dice_per_class[cls]) if train_dice_per_class[cls] else None for cls in train_dice_per_class}
        avg_train_iou_per_class = {cls: np.mean(train_iou_per_class[cls]) if train_iou_per_class[cls] else None for cls in train_iou_per_class}
        print(f"Epoch {epoch+1} | Train Per-Class Dice (present): {avg_train_dice_per_class}")
        print(f"Epoch {epoch+1} | Train Per-Class IoU (present): {avg_train_iou_per_class}")

        # Validation
        model.eval()
        val_loss, val_dice, val_iou = [], [], []
        val_dice_per_class = {cls: [] for cls in [0,2,3,4,5]}
        val_iou_per_class = {cls: [] for cls in [0,2,3,4,5]}
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                output = model(img)
                loss = loss_fn(output, mask)
                val_loss.append(loss.item())
                mean_dice, per_class_d = multiclass_dice_present(output, mask)
                mean_iou, per_class_i = multiclass_iou_present(output, mask)
                val_dice.append(mean_dice)
                val_iou.append(mean_iou)
                for cls in val_dice_per_class:
                    if cls in per_class_d:
                        val_dice_per_class[cls].append(per_class_d[cls])
                for cls in val_iou_per_class:
                    if cls in per_class_i:
                        val_iou_per_class[cls].append(per_class_i[cls])
        avg_val_dice_per_class = {cls: np.mean(val_dice_per_class[cls]) if val_dice_per_class[cls] else None for cls in val_dice_per_class}
        avg_val_iou_per_class = {cls: np.mean(val_iou_per_class[cls]) if val_iou_per_class[cls] else None for cls in val_iou_per_class}
        print(f"[Epoch {epoch+1}] Train Loss: {np.mean(train_loss):.4f}, Val Dice (present): {np.mean(val_dice):.4f}, Val IoU (present): {np.mean(val_iou):.4f}")
        print(f"[Epoch {epoch+1}] Val Per-Class Dice (present): {avg_val_dice_per_class}")
        print(f"[Epoch {epoch+1}] Val Per-Class IoU (present): {avg_val_iou_per_class}")

        # --- Evaluate on test set ---
        avg_test_dice, avg_test_dice_per_class, avg_test_iou, avg_test_iou_per_class, test_case_metrics = evaluate_loader(model, test_loader, loss_fn, split_name="test")
        print(f"[Epoch {epoch+1}] Test Dice (mean over present cancer classes): {avg_test_dice:.4f}")
        print(f"[Epoch {epoch+1}] Test Per-Class Dice (present): {avg_test_dice_per_class}")
        print(f"[Epoch {epoch+1}] Test IoU (mean over present cancer classes): {avg_test_iou:.4f}")
        print(f"[Epoch {epoch+1}] Test Per-Class IoU (present): {avg_test_iou_per_class}")

        # --- Save per-case metrics for train, val, test ---
        _, _, _, _, train_case_metrics = evaluate_loader(model, train_loader, loss_fn, split_name="train")
        _, _, _, _, val_case_metrics = evaluate_loader(model, val_loader, loss_fn, split_name="val")
        pd.DataFrame(train_case_metrics).to_csv(os.path.join(save_dir, f"train_case_metrics_epoch{epoch+1}.csv"), index=False)
        pd.DataFrame(val_case_metrics).to_csv(os.path.join(save_dir, f"val_case_metrics_epoch{epoch+1}.csv"), index=False)
        pd.DataFrame(test_case_metrics).to_csv(os.path.join(save_dir, f"test_case_metrics_epoch{epoch+1}.csv"), index=False)

        history.append({
            "epoch": epoch+1,
            "train_loss": np.mean(train_loss),
            "val_loss": np.mean(val_loss),
            "train_dice": np.mean(train_dice),
            "val_dice": np.mean(val_dice),
            "train_iou": np.mean(train_iou),
            "val_iou": np.mean(val_iou),
            "test_dice": avg_test_dice,
            "test_iou": avg_test_iou,
            **{f"train_dice_cls_{cls}": avg_train_dice_per_class[cls] for cls in avg_train_dice_per_class},
            **{f"val_dice_cls_{cls}": avg_val_dice_per_class[cls] for cls in avg_val_dice_per_class},
            **{f"test_dice_cls_{cls}": avg_test_dice_per_class[cls] for cls in avg_test_dice_per_class},
            **{f"train_iou_cls_{cls}": avg_train_iou_per_class[cls] for cls in avg_train_iou_per_class},
            **{f"val_iou_cls_{cls}": avg_val_iou_per_class[cls] for cls in avg_val_iou_per_class},
            **{f"test_iou_cls_{cls}": avg_test_iou_per_class[cls] for cls in avg_test_iou_per_class},
        })
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch+1}.pth"))
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

# --- DATASET SPLIT ---
img_root = "input/images"
mask_root = "input/picai_labels/csPCa_lesion_delineations/human_expert/resampled"
exclude_csv = "unannotated_cspca_cases.csv"
all_cases = "input/picai_labels/clinical_information/marksheet.csv"

# --- Save directory ---
save_dir = 'models/model5_trial2/'
os.makedirs(save_dir, exist_ok=True)
print(f"Model will be saved to: {save_dir}")


transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])


# Read all_cases and exclude_csv
all_cases_df = pd.read_csv(all_cases, dtype={'study_id': str})
exclude_df = pd.read_csv(exclude_csv, dtype={'study_id': str})
# Create a DataFrame consisting of all_cases except those in exclude_csv
filtered_cases_df = all_cases_df[~all_cases_df['study_id'].isin(exclude_df['study_id'])].reset_index(drop=True)

print(f"Total cases in all_cases: {len(all_cases_df)}")
print(f"Cases to exclude: {len(exclude_df)}")
print(f"Filtered cases (after exclusion): {len(filtered_cases_df)}")
filtered_cases_df.to_csv(save_dir+"/filtered_cases.csv", index=False)


# Stratified split based on "case_ISUP"
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
trainval_idx, test_idx = next(sss1.split(filtered_cases_df, filtered_cases_df["case_ISUP"]))
trainval_df = filtered_cases_df.iloc[trainval_idx].reset_index(drop=True)
test_df = filtered_cases_df.iloc[test_idx].reset_index(drop=True)

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(sss2.split(trainval_df, trainval_df["case_ISUP"]))
train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

print(f"Train cases: {len(train_df)}")
print(f"Validation cases: {len(val_df)}")
print(f"Test cases: {len(test_df)}")

train_df.to_csv(save_dir+"/train_cases.csv", index=False)
val_df.to_csv(save_dir+"/val_cases.csv", index=False)
test_df.to_csv(save_dir+"/test_cases.csv", index=False)


# --- Print class frequencies ---
train_class_freq = train_df['case_ISUP'].value_counts().to_dict()
val_class_freq = val_df['case_ISUP'].value_counts().to_dict()
test_class_freq = test_df['case_ISUP'].value_counts().to_dict()
print(f"Train ISUP class frequencies: {train_class_freq}")
print(f"Validation ISUP class frequencies: {val_class_freq}")
print(f"Test ISUP class frequencies: {test_class_freq}")


# --- Set batch size ---
BATCH_SIZE = 50
print(f"Batch size: {BATCH_SIZE}")

# --- Create datasets and dataloaders ---
train_dataset = LazyProstateSegDataset(img_root, mask_root, train_df, transform=transform)
val_dataset   = LazyProstateSegDataset(img_root, mask_root, val_df, transform=transform)
test_dataset  = LazyProstateSegDataset(img_root, mask_root, test_df, transform=transform)

print(f"Train slices: {len(train_dataset)}")
print(f"Validation slices: {len(val_dataset)}")
print(f"Test slices: {len(test_dataset)}")


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)




# --- MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = UNet(num_classes=NUM_CLASSES).to(device)
#class_weights = compute_class_weights(train_dataset)
class_weights = compute_binary_class_weights_from_df()  # Use the precomputed weights from CSV
print(f"Class weights: {class_weights.tolist()}")
class_weights = class_weights.to(device) 

# --- Loss and Optimizer ---
loss_fn = get_loss_fn(class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)




# --- RUN TRAINING ---
train_model(model, train_loader, val_loader, test_loader, optimizer, loss_fn, num_epochs=2)

# --- FINAL TESTING ---
evaluate_test(model, test_loader, loss_fn)

