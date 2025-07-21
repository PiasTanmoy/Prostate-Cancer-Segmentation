# segment_semi_supervised_documented.py
# Semi-supervised segmentation pipeline for prostate MRI using DeepLabV3+ and pseudo-labeling.
# Author: [Tanmoy Sarkar Pias]
# This script performs supervised training, pseudo-label generation, and semi-supervised training.

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

# --- Configuration ---
# Set all hyperparameters and constants for training and data loading
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
IMAGE_SIZE = 384 # Image size for resizing 
BATCH_SIZE = 8 # Batch size for training
LEARNING_RATE_S1 = 1e-4 # Learning rate for stage 1 (supervised training)
LEARNING_RATE_S2 = 5e-5 # Learning rate for stage 2 (semi-supervised training)
NUM_EPOCHS_STAGE_1 = 50  # Number of epochs for stage 1 (supervised training)
NUM_EPOCHS_STAGE_2 = 50 # Number of epochs for stage 2 (semi-supervised training)
NUM_WORKERS = 0 # Number of workers for DataLoader
PIN_MEMORY = True  # Pin memory for DataLoader 
VALIDATION_SPLIT = 0.2  # Fraction of data for validation
MIN_LESION_AREA = 50 # Minimum lesion area for post-processing
NUM_CLASSES = 6 # Number of segmentation classes

# Print configuration for reference
print(f"Using device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}, Batch size: {BATCH_SIZE}")
print(f"Learning rates: Stage 1 = {LEARNING_RATE_S1}, Stage 2 = {LEARNING_RATE_S2}")
print(f"Number of epochs: Stage 1 = {NUM_EPOCHS_STAGE_1}, Stage 2 = {NUM_EPOCHS_STAGE_2}")
print(f"Number of workers: {NUM_WORKERS}, Pin memory: {PIN_MEMORY}")
print(f"Validation split: {VALIDATION_SPLIT}, Minimum lesion area: {MIN_LESION_AREA}")
print(f"Number of classes: {NUM_CLASSES}")

# --- Directory setup ---
# Define paths for labeled and unlabeled data, output directories, and model checkpoints
ORIGINAL_DATA_DIR = "input/processed_resampled3" # Directory containing original labeled data
UNLABELED_DATA_DIR = "input/processed_incomplete_cases" # Directory containing unlabeled data
OUTPUT_DIR = "models/semisupervised_output/multi_class_trial4" # Output directory for saving models and pseudo-masks
PSEUDO_MASK_DIR = os.path.join(OUTPUT_DIR, "pseudo_masks") # Directory for saving pseudo-masks
STAGE_1_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_supervised_model.pth") # Path to save the best model from stage 1
FINAL_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_semisupervised_model.pth") # Path to save the final model after semi-supervised training
os.makedirs(PSEUDO_MASK_DIR, exist_ok=True) # Ensure pseudo-mask directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# Print directory paths for reference
print(f"Original data directory: {ORIGINAL_DATA_DIR}") 
print(f"Unlabeled data directory: {UNLABELED_DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Pseudo-mask directory: {PSEUDO_MASK_DIR}")
print(f"Stage 1 model save path: {STAGE_1_MODEL_SAVE_PATH}")
print(f"Final model save path: {FINAL_MODEL_SAVE_PATH}")




class DeepLabV3Plus(nn.Module): 
    """
    DeepLabV3+ segmentation model with configurable number of output classes.
    Uses a ResNet-101 backbone.
    """
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=True) # Load pre-trained weights
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1) # Replace final layer to match number of classes
    def forward(self, x):
        return self.model(x)['out'] # Forward pass through the model, returning logits for each class



def multiclass_dice_score(preds, targets, num_classes=6, smooth=1e-6):
    """
    Computes average Dice score across all classes.
    Args:
        preds: Model predictions (logits)
        targets: Ground truth masks
        num_classes: Number of classes
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Average Dice score
    """
    preds = torch.softmax(preds, dim=1) # Convert logits to probabilities. 
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float() # Convert targets to one-hot encoding
    dice = 0 # Initialize Dice score
    for c in range(num_classes): # Iterate over each class
        pred_flat = preds[:, c].contiguous().view(-1) # Flatten predictions for class c
        target_flat = targets_onehot[:, c].contiguous().view(-1) # Flatten targets for class c
        intersection = (pred_flat * target_flat).sum() # Compute intersection
        dice += (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth) # Compute Dice score for class c
    return dice / num_classes # Average Dice score across all classes


def binary_dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Computes Dice score for binary segmentation (foreground vs background).
    Args:
        preds: Model predictions (logits)
        targets: Ground truth masks
        threshold: Threshold for binarization
        smooth: Smoothing factor
    Returns:
        Binary Dice score
    """
    preds = torch.softmax(preds, dim=1) # Convert logits to probabilities
    preds_bin = (torch.argmax(preds, dim=1) >= 1).float() # Binarize predictions (foreground if class >= 1)
    targets_bin = (targets >= 1).float() # Binarize targets (foreground if class >= 1)
    intersection = (preds_bin * targets_bin).sum() # Compute intersection
    return (2. * intersection + smooth) / (preds_bin.sum() + targets_bin.sum() + smooth) # Compute Dice score for binary segmentation



class SemiSupPiCaiDataset(Dataset):
    """
    Custom dataset for semi-supervised segmentation.
    Handles both labeled and pseudo-labeled data, and slice sampling.
    """
    def __init__(self, base_dir, sample_ids, mask_dir, is_validation=False):
        self.base_dir = base_dir # Base directory containing modality data
        self.sample_ids = sample_ids # List of sample IDs to load
        self.mask_dir = mask_dir # Directory containing masks
        self.slice_infos = [] # List to store (sample_idx, slice_idx) tuples for each slice
        
        if not self.sample_ids: return # If no sample IDs provided, return empty dataset
        first_mask_path = os.path.join(mask_dir, f'{self.sample_ids[0]}.npy') # Path to the first mask file
        
        if not os.path.exists(first_mask_path): # If the first mask file does not exist, try to find any existing mask
            for s_id in self.sample_ids: # Iterate through sample IDs to find an existing mask
                potential_path = os.path.join(mask_dir, f'{s_id}.npy') # Path to potential mask file
                if os.path.exists(potential_path): # If the potential mask file exists, use it
                    first_mask_path = potential_path # Update first_mask_path to the found mask
                    break
            else:
                return
            
        data_shape = np.load(first_mask_path).shape # Load shape of the first mask to determine slice axis
        self.slice_axis = np.argmin(data_shape) # Determine the slice axis based on the smallest dimension
        desc = "Finding validation slices" if is_validation else "Finding training slices" 

        for sample_idx, sample_id in enumerate(tqdm(self.sample_ids, desc=desc)): # Iterate through sample IDs to find slices
            
            mask_path = os.path.join(self.mask_dir, f'{sample_id}.npy') # Path to the mask file for the current sample
            
            if not os.path.exists(mask_path): continue # If the mask file does not exist, skip this sample
            
            mask_3d = np.load(mask_path) # Load the 3D mask for the current sample
            num_slices = mask_3d.shape[self.slice_axis] # Get the number of slices along the slice axis
            
            if is_validation:
                for slice_idx in range(num_slices):
                    self.slice_infos.append((sample_idx, slice_idx)) # sample_idx is the index of the sample, slice_idx is the index of the slice
            else:
                # For training, we sample positive and negative slices
                pos_slices = [i for i in range(num_slices) if np.sum(np.take(mask_3d, i, self.slice_axis) >= 1) > 0] # Get positive slices where mask (pixel sum not 0) is present
                neg_slices = [i for i in range(num_slices) if np.sum(np.take(mask_3d, i, self.slice_axis) >= 1) == 0] # Get negative slices where mask (pixel all 0) is absent
                
                num_pos = len(pos_slices) # Number of positive slices
                for slice_idx in pos_slices: 
                    self.slice_infos.append((sample_idx, slice_idx)) # Append positive slices to slice_infos
                
                if neg_slices:
                    neg_samples = random.sample(neg_slices, min(num_pos, len(neg_slices))) # Randomly sample negative slices, ensuring we have the same number as positive slices
                    for slice_idx in neg_samples: 
                        self.slice_infos.append((sample_idx, slice_idx))

        random.shuffle(self.slice_infos) # Shuffle the slice infos to randomize the order of slices

    def __len__(self):
        return len(self.slice_infos) # Return the number of slices in the dataset
    
    def __getitem__(self, idx):
        sample_idx, slice_idx = self.slice_infos[idx] # Get the sample index and slice index for the given idx
        sample_id = self.sample_ids[sample_idx] # Get the ORIGINAL sample ID corresponding to the sample index
        modalities = [np.load(os.path.join(self.base_dir, m, f'{sample_id}.npy')) for m in ['t2w', 'adc', 'hbv']] # Load modalities for the sample
        img_3d = np.stack(modalities, axis=-1) # Combines the three 3D modality arrays into a single 4D array, with the last dimension representing the modality channel.
        mask_path = os.path.join(self.mask_dir, f'{sample_id}.npy') # Path to the mask file for the sample
        mask_3d = np.load(mask_path) # Load the 3D mask for the sample
        image_slice = np.take(img_3d, slice_idx, axis=self.slice_axis) # Extract the slice from the 3D image for all 3 channels
        mask_slice = np.take(mask_3d, slice_idx, axis=self.slice_axis) # Extract the corresponding mask slice
        mask_slice = mask_slice.astype(np.int64) # Ensures the mask is in the correct integer format for PyTorch. 
        return image_slice, mask_slice # Returns the 2D image slice (with all modalities as channels) and its corresponding mask slice.



class AugmentationWrapper(Dataset):
    """
    Wrapper for applying augmentations to the dataset.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_np, mask_np = self.dataset[idx] # Load the image and mask from the dataset

        if self.transform:
            augmented = self.transform(image=image_np.astype(np.float32), mask=mask_np)
            image_np, mask_np = augmented['image'], augmented['mask']

        for i in range(image_np.shape[2]): # Iterate over each channel in the image which is 3
            channel = image_np[:, :, i] # Extract the 2D channel.
            non_zero = channel[channel > 1e-6] # Find all non-zero pixels (to avoid background).
            if non_zero.size > 0:
                p1, p99 = np.percentile(non_zero, 1), np.percentile(non_zero, 99)
                channel = np.clip(channel, p1, p99) # Clip all values in the channel to be within [p1, p99]. This reduces the effect of extreme outliers, making the normalization more robust.

            min_val, max_val = channel.min(), channel.max() 
            image_np[:, :, i] = (channel - min_val) / (max_val - min_val) if max_val > min_val else 0 # Normalize the channel to [0, 1] range. If max_val equals min_val, set the channel to 0 to avoid division by zero.

        image = torch.from_numpy(image_np.transpose(2, 0, 1)).float() # Convert the image to a PyTorch tensor, changing the shape from (H, W, C) to (C, H, W), and cast to float.
        mask = torch.from_numpy(mask_np).long() # Convert the mask to a PyTorch tensor and cast to long type for segmentation tasks.

        return image, mask


def train_fn(loader, model, loss_fn, optimizer, scaler, device):
    """
    Training function for the model.
    """
    model.train()
    loop = tqdm(loader, desc="Training")
    for data, targets in loop:
        data, targets = data.to(device), targets.to(device)

        with torch.amp.autocast(device_type=device.split(':')[0], enabled=(device=="cuda")): # Enable mixed precision training if using CUDA
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad() # Zero the gradients before backpropagation
        scaler.scale(loss).backward() # Scale the loss for mixed precision
        scaler.step(optimizer) # Update the model parameters
        scaler.update() # Update the scaler for the next iteration
        loop.set_postfix(loss=loss.item()) # Update the progress bar with the current loss

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
                ref_vol = np.load(ref_vol_path) # Load the reference volume (T2-weighted image)
                slice_axis = np.argmin(ref_vol.shape) # Determine the slice axis based on the smallest dimension
                pred_volume = np.zeros_like(ref_vol, dtype=np.uint8) # Prepares an empty array to store the predicted mask for each slice
                
                for slice_idx in range(ref_vol.shape[slice_axis]): # Loops through every 2D slice along the chosen axis.
                    modalities = [np.load(os.path.join(base_dir, m, f'{patient_id}.npy')) for m in ['t2w', 'adc', 'hbv']] # Loads the three MRI modalities (t2w, adc, hbv) for the patient.
                    image_slice_np = np.stack([np.take(vol, slice_idx, axis=slice_axis) for vol in modalities], axis=-1) # Extracts the corresponding 2D slice from each modality and stacks them into a 3-channel image.
                    eval_transform.dataset = [(image_slice_np, np.zeros_like(image_slice_np[...,0]))] # Sets the wrapper’s dataset to the current slice (mask is a dummy, not used).
                    image_tensor, _ = eval_transform[0] # Applies the evaluation transform to get the image tensor. Applies resizing and normalization.
                    image_tensor = image_tensor.unsqueeze(0).to(device)

                    with torch.amp.autocast(device_type=device.split(':')[0], enabled=(device=="cuda")):
                        pred_logits = model(image_tensor) 
                        pred_class = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1).squeeze().cpu().numpy().astype(np.uint8) # Gets the predicted class for each pixel in the slice by applying softmax and argmax.
                    
                    original_dims = (ref_vol.shape[1], ref_vol.shape[2])
                    resizer = A.Resize(height=original_dims[0], width=original_dims[1], interpolation=0) 
                    pred_resized = resizer(image=pred_class)['image']# Resizes the predicted mask back to the original slice dimensions.

                    pred_processed = remove_small_lesions_multiclass(pred_resized, min_lesion_area, num_classes)
                    slicer = [slice(None)] * pred_volume.ndim # Creates a slicer for the 3D volume.
                    slicer[slice_axis] = slice_idx # Sets the slice index for the current slice.
                    pred_volume[tuple(slicer)] = pred_processed # Assigns the processed mask back to the corresponding slice in the 3D volume.

                np.save(os.path.join(out_dir, f"{patient_id}.npy"), pred_volume) # Saves the full 3D pseudo-labeled mask for the patient.

            except Exception as e:
                print(f"Warning: Could not pseudo-label patient {patient_id}. Error: {e}")

    model.train()


def remove_small_lesions_multiclass(mask_np, min_size, num_classes):
    out = np.copy(mask_np) # predicted mask
    for c in range(1, num_classes):
        binary = (mask_np == c).astype(np.uint8)
        labeled_array, num_features = ndi.label(binary) # labeled_array assigns a unique integer to each connected region, num_features is the number of connected regions found.
        if num_features == 0: # If there are no regions for this class, skip to the next class.
            continue
        component_sizes = np.bincount(labeled_array.ravel()) # Counts the number of pixels in each connected component (region).
        large_enough = component_sizes > min_size  # Creates a boolean array indicating which components are larger than the minimum size.
        large_enough[0] = False # Ensures the background label (0) is not considered a valid region.
        keep = large_enough[labeled_array] # Creates a mask of the same shape as the image, True where the region is large enough, False otherwise.
        out[(binary == 1) & (~keep)] = 0 # Sets pixels belonging to small regions (not "keep") back to 0 (background) in the output mask.

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

# ORIGINAL_DATA_DIR = input/processed_resampled3
# Lists all mask files in the labeled data directory, strips the .npy extension, and sorts the IDs.
# This is just to extract the IDs of the labeled data.
all_original_ids = sorted([f.replace('.npy', '') for f in os.listdir(os.path.join(ORIGINAL_DATA_DIR, 'mask')) if f.endswith('.npy')])

random.seed(42)
random.shuffle(all_original_ids)

# train 0.8 and validation 0.2 split
split_idx = int(len(all_original_ids) * (1 - VALIDATION_SPLIT))
original_train_ids, val_ids = all_original_ids[:split_idx], all_original_ids[split_idx:]
print(f"Labeled Data Split: {len(original_train_ids)} training, {len(val_ids)} validation patients.")

# IMAGE_SIZE = 384
train_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),  # Resize images to fixed size
    A.Rotate(limit=15, p=0.5), # Random rotation within 15 degrees
    A.HorizontalFlip(p=0.5), # Random horizontal flip
    A.VerticalFlip(p=0.1), # Random vertical flip
    A.RandomBrightnessContrast(p=0.3), # Random brightness and contrast adjustment
    A.GaussNoise(p=0.2) # Random Gaussian noise
])

val_transform = A.Compose([A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE)])

# Create datasets for training and validation
train_base_s1 = SemiSupPiCaiDataset(ORIGINAL_DATA_DIR, original_train_ids, mask_dir=os.path.join(ORIGINAL_DATA_DIR, 'mask'))
train_dataset_s1 = AugmentationWrapper(train_base_s1, transform=train_transform)
train_loader_s1 = DataLoader(train_dataset_s1, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

val_base_s1 = SemiSupPiCaiDataset(ORIGINAL_DATA_DIR, val_ids, mask_dir=os.path.join(ORIGINAL_DATA_DIR, 'mask'), is_validation=True)
val_dataset_s1 = AugmentationWrapper(val_base_s1, transform=val_transform)

val_base_s1_balanced = SemiSupPiCaiDataset(ORIGINAL_DATA_DIR, val_ids, mask_dir=os.path.join(ORIGINAL_DATA_DIR, 'mask'), is_validation=False)
val_dataset_s1_balanced = AugmentationWrapper(val_base_s1_balanced, transform=val_transform)
val_loader_s1_balanced = DataLoader(val_dataset_s1_balanced, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def TRAIN():
    # --- STAGE 1: Supervised Training ---
    print("\n--- STAGE 1: Starting Supervised Training ---")
    best_val_dice = -1.0 # Initialize best validation Dice score
    patience = 10 # Number of epochs to wait for improvement before early stopping
    patience_counter = 0 # Counter for early stopping patience
    
    for epoch in range(NUM_EPOCHS_STAGE_1): # Loop through epochs
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
    # UNLABELED_DATA_DIR = "input/processed_incomplete_cases" 
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
