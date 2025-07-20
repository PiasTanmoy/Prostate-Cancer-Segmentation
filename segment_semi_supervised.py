import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy.ndimage as ndi
from torchvision import models
from sklearn.metrics import roc_auc_score, average_precision_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 384
BATCH_SIZE = 8
LEARNING_RATE_S1 = 1e-4
LEARNING_RATE_S2 = 5e-5
NUM_EPOCHS_STAGE_1 = 50
NUM_EPOCHS_STAGE_2 = 50
NUM_WORKERS = 0
PIN_MEMORY = True
VALIDATION_SPLIT = 0.2
MIN_LESION_AREA = 50
NUM_CLASSES = 6

print(f"Using device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}, Batch size: {BATCH_SIZE}")
print(f"Learning rates: Stage 1 = {LEARNING_RATE_S1}, Stage 2 = {LEARNING_RATE_S2}")
print(f"Number of epochs: Stage 1 = {NUM_EPOCHS_STAGE_1}, Stage 2 = {NUM_EPOCHS_STAGE_2}")
print(f"Number of workers: {NUM_WORKERS}, Pin memory: {PIN_MEMORY}")
print(f"Validation split: {VALIDATION_SPLIT}, Minimum lesion area: {MIN_LESION_AREA}")
print(f"Number of classes: {NUM_CLASSES}")


ORIGINAL_DATA_DIR = "input/processed_resampled3"
UNLABELED_DATA_DIR = "input/processed_incomplete_cases"
OUTPUT_DIR = "models/semisupervised_output/multi_class_trial4"
PSEUDO_MASK_DIR = os.path.join(OUTPUT_DIR, "pseudo_masks")
STAGE_1_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_supervised_model.pth")
FINAL_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_semisupervised_model.pth")
os.makedirs(PSEUDO_MASK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Original data directory: {ORIGINAL_DATA_DIR}")
print(f"Unlabeled data directory: {UNLABELED_DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Pseudo-mask directory: {PSEUDO_MASK_DIR}")
print(f"Stage 1 model save path: {STAGE_1_MODEL_SAVE_PATH}")
print(f"Final model save path: {FINAL_MODEL_SAVE_PATH}")

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    def forward(self, x):
        return self.model(x)['out']



def multiclass_dice_score(preds, targets, num_classes=6, smooth=1e-6):
    preds = torch.softmax(preds, dim=1)
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dice = 0
    for c in range(num_classes):
        pred_flat = preds[:, c].contiguous().view(-1)
        target_flat = targets_onehot[:, c].contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice += (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice / num_classes


def binary_dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = torch.softmax(preds, dim=1)
    preds_bin = (torch.argmax(preds, dim=1) >= 1).float()
    targets_bin = (targets >= 1).float()
    intersection = (preds_bin * targets_bin).sum()
    return (2. * intersection + smooth) / (preds_bin.sum() + targets_bin.sum() + smooth)



class SemiSupPiCaiDataset(Dataset):
    def __init__(self, base_dir, sample_ids, mask_dir, is_validation=False):
        self.base_dir = base_dir
        self.sample_ids = sample_ids
        self.mask_dir = mask_dir
        self.slice_infos = []
        if not self.sample_ids: return
        first_mask_path = os.path.join(mask_dir, f'{self.sample_ids[0]}.npy')
        if not os.path.exists(first_mask_path):
            for s_id in self.sample_ids:
                potential_path = os.path.join(mask_dir, f'{s_id}.npy')
                if os.path.exists(potential_path):
                    first_mask_path = potential_path
                    break
            else:
                return
        data_shape = np.load(first_mask_path).shape
        self.slice_axis = np.argmin(data_shape)
        desc = "Finding validation slices" if is_validation else "Finding training slices"
        for sample_idx, sample_id in enumerate(tqdm(self.sample_ids, desc=desc)):
            mask_path = os.path.join(self.mask_dir, f'{sample_id}.npy')
            if not os.path.exists(mask_path):
                continue
            mask_3d = np.load(mask_path)
            num_slices = mask_3d.shape[self.slice_axis]
            if is_validation:
                for slice_idx in range(num_slices):
                    self.slice_infos.append((sample_idx, slice_idx))
            else:
                pos_slices = [i for i in range(num_slices) if np.sum(np.take(mask_3d, i, self.slice_axis) >= 1) > 0]
                neg_slices = [i for i in range(num_slices) if np.sum(np.take(mask_3d, i, self.slice_axis) >= 1) == 0]
                num_pos = len(pos_slices)
                for slice_idx in pos_slices: self.slice_infos.append((sample_idx, slice_idx))
                if neg_slices:
                    neg_samples = random.sample(neg_slices, min(num_pos, len(neg_slices)))
                    for slice_idx in neg_samples: self.slice_infos.append((sample_idx, slice_idx))
        random.shuffle(self.slice_infos)
    def __len__(self):
        return len(self.slice_infos)
    def __getitem__(self, idx):
        sample_idx, slice_idx = self.slice_infos[idx]
        sample_id = self.sample_ids[sample_idx]
        modalities = [np.load(os.path.join(self.base_dir, m, f'{sample_id}.npy')) for m in ['t2w', 'adc', 'hbv']]
        img_3d = np.stack(modalities, axis=-1)
        mask_path = os.path.join(self.mask_dir, f'{sample_id}.npy')
        mask_3d = np.load(mask_path)
        image_slice = np.take(img_3d, slice_idx, axis=self.slice_axis)
        mask_slice = np.take(mask_3d, slice_idx, axis=self.slice_axis)
        mask_slice = mask_slice.astype(np.int64)
        return image_slice, mask_slice

class AugmentationWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image_np, mask_np = self.dataset[idx]
        if self.transform:
            augmented = self.transform(image=image_np.astype(np.float32), mask=mask_np)
            image_np, mask_np = augmented['image'], augmented['mask']
        for i in range(image_np.shape[2]):
            channel = image_np[:, :, i]
            non_zero = channel[channel > 1e-6]
            if non_zero.size > 0:
                p1, p99 = np.percentile(non_zero, 1), np.percentile(non_zero, 99)
                channel = np.clip(channel, p1, p99)
            min_val, max_val = channel.min(), channel.max()
            image_np[:, :, i] = (channel - min_val) / (max_val - min_val) if max_val > min_val else 0
        image = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask_np).long()
        return image, mask


def train_fn(loader, model, loss_fn, optimizer, scaler, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    for data, targets in loop:
        data, targets = data.to(device), targets.to(device)
        with torch.amp.autocast(device_type=device.split(':')[0], enabled=(device=="cuda")):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

def check_accuracy(loader, model, device, num_classes=6):
    model.eval()
    multiclass_dice_sum, binary_dice_sum, count = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.split(':')[0], enabled=(device=="cuda")):
                preds = model(x)
            multiclass_dice_sum += multiclass_dice_score(preds, y, num_classes=num_classes).item()
            binary_dice_sum += binary_dice_score(preds, y).item()
            count += 1
    model.train()
    return multiclass_dice_sum / count, binary_dice_sum / count

def generate_pseudo_labels(unlabeled_ids, base_dir, out_dir, model, device, image_size, num_classes, min_lesion_area):
    model.eval()
    eval_transform = AugmentationWrapper(dataset=None, transform=A.Compose([A.Resize(height=image_size, width=image_size)]))
    with torch.no_grad():
        for patient_id in tqdm(unlabeled_ids, desc="Generating Pseudo-Masks"):
            try:
                ref_vol_path = os.path.join(base_dir, 't2w', f'{patient_id}.npy')
                if not os.path.exists(ref_vol_path): continue
                ref_vol = np.load(ref_vol_path)
                slice_axis = np.argmin(ref_vol.shape)
                pred_volume = np.zeros_like(ref_vol, dtype=np.uint8)
                for slice_idx in range(ref_vol.shape[slice_axis]):
                    modalities = [np.load(os.path.join(base_dir, m, f'{patient_id}.npy')) for m in ['t2w', 'adc', 'hbv']]
                    image_slice_np = np.stack([np.take(vol, slice_idx, axis=slice_axis) for vol in modalities], axis=-1)
                    eval_transform.dataset = [(image_slice_np, np.zeros_like(image_slice_np[...,0]))]
                    image_tensor, _ = eval_transform[0]
                    image_tensor = image_tensor.unsqueeze(0).to(device)
                    with torch.amp.autocast(device_type=device.split(':')[0], enabled=(device=="cuda")):
                        pred_logits = model(image_tensor)
                        pred_class = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1).squeeze().cpu().numpy().astype(np.uint8)
                    original_dims = (ref_vol.shape[1], ref_vol.shape[2])
                    resizer = A.Resize(height=original_dims[0], width=original_dims[1], interpolation=0)
                    pred_resized = resizer(image=pred_class)['image']
                    pred_processed = remove_small_lesions_multiclass(pred_resized, min_lesion_area, num_classes)
                    slicer = [slice(None)] * pred_volume.ndim
                    slicer[slice_axis] = slice_idx
                    pred_volume[tuple(slicer)] = pred_processed
                np.save(os.path.join(out_dir, f"{patient_id}.npy"), pred_volume)
            except Exception as e:
                print(f"Warning: Could not pseudo-label patient {patient_id}. Error: {e}")
    model.train()

def remove_small_lesions_multiclass(mask_np, min_size, num_classes):
    out = np.copy(mask_np)
    for c in range(1, num_classes):
        binary = (mask_np == c).astype(np.uint8)
        labeled_array, num_features = ndi.label(binary)
        if num_features == 0:
            continue
        component_sizes = np.bincount(labeled_array.ravel())
        large_enough = component_sizes > min_size
        large_enough[0] = False
        keep = large_enough[labeled_array]
        out[(binary == 1) & (~keep)] = 0
    return out


# --- Model Initialization ---
print("\n--- Model Initialization ---")
model = DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_S1)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))
best_val_dice = -1.0


# --- Data Preparation ---
print("\n--- Data Preparation ---")
all_original_ids = sorted([f.replace('.npy', '') for f in os.listdir(os.path.join(ORIGINAL_DATA_DIR, 'mask')) if f.endswith('.npy')])
random.seed(42); random.shuffle(all_original_ids)
split_idx = int(len(all_original_ids) * (1 - VALIDATION_SPLIT))
original_train_ids, val_ids = all_original_ids[:split_idx], all_original_ids[split_idx:]
print(f"Labeled Data Split: {len(original_train_ids)} training, {len(val_ids)} validation patients.")

train_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE), A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.1),
    A.RandomBrightnessContrast(p=0.3), A.GaussNoise(p=0.2)
])
val_transform = A.Compose([A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE)])

train_base_s1 = SemiSupPiCaiDataset(ORIGINAL_DATA_DIR, original_train_ids, mask_dir=os.path.join(ORIGINAL_DATA_DIR, 'mask'))
val_base_s1 = SemiSupPiCaiDataset(ORIGINAL_DATA_DIR, val_ids, mask_dir=os.path.join(ORIGINAL_DATA_DIR, 'mask'), is_validation=True)

train_dataset_s1 = AugmentationWrapper(train_base_s1, transform=train_transform)
val_dataset_s1 = AugmentationWrapper(val_base_s1, transform=val_transform)

train_loader_s1 = DataLoader(train_dataset_s1, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_base_s1_balanced = SemiSupPiCaiDataset(ORIGINAL_DATA_DIR, val_ids, mask_dir=os.path.join(ORIGINAL_DATA_DIR, 'mask'), is_validation=False)
val_dataset_s1_balanced = AugmentationWrapper(val_base_s1_balanced, transform=val_transform)
val_loader_s1_balanced = DataLoader(val_dataset_s1_balanced, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def TRAIN():
    # --- STAGE 1: Supervised Training ---
    print("\n--- STAGE 1: Starting Supervised Training ---")
    best_val_dice = -1.0
    patience = 10
    patience_counter = 0
    for epoch in range(NUM_EPOCHS_STAGE_1):
        print(f"\n--- Stage 1, Epoch {epoch+1}/{NUM_EPOCHS_STAGE_1} ---")
        train_fn(train_loader_s1, model, loss_fn, optimizer, scaler, DEVICE)
        multiclass_dice, binary_dice = check_accuracy(val_loader_s1_balanced, model, DEVICE, num_classes=NUM_CLASSES)
        print(f"Validation Multiclass Dice: {multiclass_dice:.4f}")
        print(f"Validation Binary Dice: {binary_dice:.4f}")
        if multiclass_dice > best_val_dice:
            best_val_dice = multiclass_dice
            torch.save(model.state_dict(), STAGE_1_MODEL_SAVE_PATH)
            print(f"==> New best Stage 1 model saved with Multiclass Dice: {best_val_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Early stop patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print(f"\nTraining finished. Best model at: {STAGE_1_MODEL_SAVE_PATH if os.path.exists(STAGE_1_MODEL_SAVE_PATH) else 'not saved'}")


def pred_segment():
    print("\n--- Starting Prediction and Segmentation ---")
    # --- Pseudo-Label Generation ---
    print("\n--- Generating Pseudo-Labels for Stage 2 ---")
    model.load_state_dict(torch.load(STAGE_1_MODEL_SAVE_PATH))
    unlabeled_ids = sorted([f.replace('.npy', '') for f in os.listdir(os.path.join(UNLABELED_DATA_DIR, 't2w'))])
    generate_pseudo_labels(unlabeled_ids, UNLABELED_DATA_DIR, PSEUDO_MASK_DIR, model, DEVICE, IMAGE_SIZE, NUM_CLASSES, MIN_LESION_AREA)

def semi_supervised_train():
    print("\n--- Starting Semi-Supervised Training ---")
    # --- STAGE 2: Semi-Supervised Training ---
    print("\n--- STAGE 2: Starting Semi-Supervised Training ---")
    model.load_state_dict(torch.load(STAGE_1_MODEL_SAVE_PATH))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_S2)

    unlabeled_ids = sorted([f.replace('.npy', '') for f in os.listdir(os.path.join(UNLABELED_DATA_DIR, 't2w'))])
    pseudo_train_base = SemiSupPiCaiDataset(UNLABELED_DATA_DIR, unlabeled_ids, mask_dir=PSEUDO_MASK_DIR)
    if len(pseudo_train_base) > 0:
        combined_train_dataset = ConcatDataset([train_base_s1, pseudo_train_base])
        print(f"Combining {len(train_base_s1)} labeled slices with {len(pseudo_train_base)} pseudo-labeled slices.")
    else:
        print("No pseudo-labeled slices were generated. Proceeding with only labeled data for Stage 2.")
        combined_train_dataset = train_base_s1

    train_dataset_s2 = AugmentationWrapper(combined_train_dataset, transform=train_transform)
    train_loader_s2 = DataLoader(train_dataset_s2, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    best_val_dice = -1.0
    patience = 10
    patience_counter = 0
    for epoch in range(NUM_EPOCHS_STAGE_2):
        print(f"\n--- Stage 2, Epoch {epoch+1}/{NUM_EPOCHS_STAGE_2} ---")
        train_fn(train_loader_s2, model, loss_fn, optimizer, scaler, DEVICE)
        multiclass_dice, binary_dice = check_accuracy(val_loader_s1_balanced, model, DEVICE, num_classes=NUM_CLASSES)
        print(f"Validation Multiclass Dice: {multiclass_dice:.4f}")
        print(f"Validation Binary Dice: {binary_dice:.4f}")
        if multiclass_dice > best_val_dice:
            best_val_dice = multiclass_dice
            torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
            print(f"==> New best Stage 2 model saved with Multiclass Dice: {best_val_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Early stop patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
    print(f"\nTraining finished. Best model at: {FINAL_MODEL_SAVE_PATH if os.path.exists(FINAL_MODEL_SAVE_PATH) else STAGE_1_MODEL_SAVE_PATH}")


print("Current time:", os.popen('date').read().strip())


# model.load_state_dict(torch.load("models/semisupervised_output/multi_class_trial2/best_supervised_model_e50.pth"))
# print("model loaded from:", "models/semisupervised_output/multi_class_trial2/best_supervised_model_e50.pth")


print("Current time:", os.popen('date').read().strip())
print("TRAIN()")
TRAIN()


print("Current time:", os.popen('date').read().strip())
print("pred_segment()")
pred_segment()


print("Current time:", os.popen('date').read().strip())
print("semi_supervised_train()")
semi_supervised_train()

print("Current time:", os.popen('date').read().strip())
