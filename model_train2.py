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
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from torch.utils.data import Dataset
import torch

import torch.nn.functional as F

max_h, max_w = 1024, 1024  # Set your desired max dimensions here
TARGET_H, TARGET_W = 256, 256   # wanted height & width

def pad_to_size(tensor, target_h, target_w):
    _, h, w = tensor.shape
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)


class LazyProstateSegDataset(Dataset):
    def __init__(self, img_root, mask_root, exclude_csv, transform=None):
        self.transform = transform
        self.entries = []

        # Read excluded study IDs
        excluded = set(pd.read_csv(exclude_csv)['study_id'].astype(str).tolist())

        for patient_folder in sorted(os.listdir(img_root)):
            patient_path = os.path.join(img_root, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            for f in os.listdir(patient_path):
                if not f.endswith('_t2w.mha'):
                    continue

                study_id = f.replace('_t2w.mha', '')
                if study_id in excluded:
                    continue

                img_path = os.path.join(patient_path, f)
                mask_path = os.path.join(mask_root, study_id + '.nii.gz')
                if not os.path.exists(mask_path):
                    continue

                try:
                    img_header = sitk.ReadImage(img_path)
                    img_size = img_header.GetSize()  # (W, H, D)
                    depth = img_size[2]

                    for z in range(depth):
                        self.entries.append({
                            'study_id': study_id,
                            'img_path': img_path,
                            'mask_path': mask_path,
                            'slice_idx': z
                        })

                except Exception as e:
                    print(f"[Warning] Skipping {study_id} due to header load error: {e}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Read and extract 2D slice only
        img_volume = sitk.GetArrayFromImage(sitk.ReadImage(entry['img_path']))  # shape: [D, H, W]
        mask_volume = nib.load(entry['mask_path']).get_fdata()
        mask_volume = np.transpose(mask_volume, (2, 0, 1)) if mask_volume.shape != img_volume.shape else mask_volume

        z = entry['slice_idx']
        img_slice = img_volume[z].astype(np.float32)
        mask_slice = mask_volume[z].astype(np.float32)

        # Normalize
        img_slice = (img_slice - np.mean(img_slice)) / (np.std(img_slice) + 1e-6)

        # Remove channel dimension for cv2.resize
        img_slice = img_slice  # shape: [H, W]
        mask_slice = mask_slice  # shape: [H, W]

        img_slice  = cv2.resize(img_slice,  (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        mask_slice = cv2.resize(mask_slice, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

        # Add channel dimension back
        img_slice = np.expand_dims(img_slice, axis=0)
        mask_slice = np.expand_dims(mask_slice, axis=0)

        img_tensor = torch.tensor(img_slice, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.float32)


        # img_tensor = pad_to_size(img_tensor, max_h, max_w)
        # mask_tensor = pad_to_size(mask_tensor, max_h, max_w)

        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            mask_tensor = self.transform(mask_tensor)

        return img_tensor, mask_tensor


# ------- Model -------
class UNet(nn.Module):
    def __init__(self):
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
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return torch.sigmoid(self.out(d2))



# ------- Dice Loss -------
def dice_loss(pred, target, smooth=1.):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))



# ------- Training -------
def train_model(model, dataloader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for img, mask in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = dice_loss(output, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(dataloader):.4f}")



# --- CONFIG ---
base_path = 'models/model2/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
save_dir = base_path
os.makedirs(save_dir, exist_ok=True)



# --- METRICS ---
def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).cpu().numpy()

def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).clamp(0,1).sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).cpu().numpy()

def flat_metrics(pred, target):
    pred_flat = pred.view(-1).cpu().numpy() > 0.5
    target_flat = target.view(-1).cpu().numpy()
    return {
        "precision": precision_score(target_flat, pred_flat, zero_division=0),
        "recall": recall_score(target_flat, pred_flat, zero_division=0),
        "f1": f1_score(target_flat, pred_flat, zero_division=0),
        "iou": jaccard_score(target_flat, pred_flat, zero_division=0)
    }


# --- TRAINING ---
def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    tot_pos, tot_neg = 0, 0
    with torch.no_grad():
        for _, masks in train_loader:                     # masks ∈ [B,1,H,W] or [B,1,H,W,D]
            pos = masks.sum().item()
            neg = masks.numel() - pos
            tot_pos += pos
            tot_neg += neg

    # avoid div-by-zero
    pos_weight = torch.tensor([tot_neg / (tot_pos + 1e-6)], device=device)
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"⚖️  Using balanced BCE loss — pos_weight = {pos_weight.item():.2f}")

    
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_dice = [], []

        for batch_idx, (img, mask) in enumerate(train_loader):
            img, mask = img.to(device), mask.to(device)
            
            # print frequency of mask values
            unique, counts = torch.unique(mask, return_counts=True)
            print(f"Batch {batch_idx+1}: Mask unique values: {dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))}")
            
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                prob = torch.sigmoid(output)
                batch_dice = dice_score(prob, mask)
                train_dice.extend(dice_score(prob, mask))
                train_loss.append(loss.item())

            # ✅ Per-batch logging
            print(f"[Epoch {epoch+1:02d}, Batch {batch_idx+1:03d}] "
              f"Loss: {loss.item():.4f}, Dice: {np.mean(batch_dice):.4f}")

        # Validation
        model.eval()
        val_loss, val_dice = [], []
        val_metrics = []

        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                output = model(img)
                loss = criterion(output, mask)
                prob = torch.sigmoid(output)
                val_dice.extend(dice_score(prob, mask))
                val_loss.append(loss.item())
                #val_metrics.append(flat_metrics(prob, mask))

        # Aggregate results
        val_metrics_df = pd.DataFrame(val_metrics).mean().to_dict()
        epoch_summary = {
            "epoch": epoch+1,
            "train_loss": np.mean(train_loss),
            "train_dice": np.mean(train_dice),
            "val_loss": np.mean(val_loss),
            "val_dice": np.mean(val_dice),
            **val_metrics_df
        }
        history.append(epoch_summary)

        print(f"[Epoch {epoch+1}] "
              f"Train Loss: {epoch_summary['train_loss']:.4f}, "
              f"Val Dice: {epoch_summary['val_dice']:.4f}, "
              f"Val IoU: {epoch_summary['iou']:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch+1}.pth"))

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_log.csv"), index=False)


# --- TESTING ---
def evaluate_test(model, test_loader):
    model.eval()
    test_dice, test_metrics = [], []

    with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            prob = torch.sigmoid(output)
            test_dice.extend(dice_score(prob, mask))
            #test_metrics.append(flat_metrics(prob, mask))

    test_summary = pd.DataFrame(test_metrics).mean()
    test_summary["dice"] = np.mean(test_dice)
    test_summary.to_csv(os.path.join(save_dir, "test_metrics.csv"))
    print("\n📊 Test Performance:")
    print(test_summary.round(4))




# --- DATASET SPLIT ---
img_root = "input/images"
mask_root = "input/picai_labels/csPCa_lesion_delineations/human_expert/resampled"
exclude_csv = "unannotated_cspca_cases.csv"

transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
dataset = LazyProstateSegDataset(img_root, mask_root, exclude_csv, transform=transform)
print(f"Total valid slices: {len(dataset)}")

indices = list(range(len(dataset)))
train_ids, test_ids = train_test_split(indices, test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.15, random_state=42)

train_loader = DataLoader(Subset(dataset, train_ids), batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(Subset(dataset, val_ids), batch_size=2, shuffle=False)
test_loader = DataLoader(Subset(dataset, test_ids), batch_size=2, shuffle=False)

# --- MODEL SETUP ---
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- RUN TRAINING ---
train_model(model, train_loader, val_loader, optimizer, num_epochs=2)

# --- FINAL TESTING ---
evaluate_test(model, test_loader)

