
import os
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from sklearn.utils import class_weight
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence


# 1. Define Dice Coefficient and Dice Loss for Binary Segmentation
def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

def dice_loss(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred))
    return 1 - (2. * intersection + K.epsilon()) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + K.epsilon())

class ProstateDataset(Sequence):
    def __init__(self, patient_list, batch_size, target_size=(256, 256)):  # Fixed size 256x256
        self.patient_list = patient_list
        self.batch_size = batch_size
        self.target_size = target_size
        self.slices = self._build_index()

    def _build_index(self):
        return [(p_idx, s_idx) for p_idx, patient in enumerate(self.patient_list) for s_idx in range(len(patient))]

    def __len__(self):
        return int(np.ceil(len(self.slices) / self.batch_size))

    def __getitem__(self, idx):
        batch_slices = self.slices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []
        for p_id, s_id in batch_slices:
            img, mask = self.patient_list[p_id][s_id]
            
            # **Safe resizing using cv2**
            img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)  # Nearest neighbor for masks
            
            # **Normalize image and process mask**
            X.append(np.expand_dims(img_resized, -1) / 255.0)  # Normalize to [0,1]
            Y.append(np.expand_dims((mask_resized > 0.5).astype(np.float32), -1))  # Convert mask to binary format
            
        return np.array(X), np.array(Y)

# 3. Load Data Patient-wise
def load_patientwise_data(images_dir, labels_dir):
    patient_data = []
    for folder in sorted(os.listdir(images_dir)):
        folder_path = os.path.join(images_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        t2w_file = [f for f in os.listdir(folder_path) if f.endswith('_t2w.mha')]
        if not t2w_file:
            continue
        fname = t2w_file[0]
        pid, sid, _ = fname.replace(".mha", "").split("_")
        t2w_path = os.path.join(folder_path, fname)
        label_name = f"{pid}_{sid}.nii.gz"
        label_path = os.path.join(labels_dir, label_name)
        if not os.path.exists(label_path):
            continue
        img_volume = sitk.GetArrayFromImage(sitk.ReadImage(t2w_path))
        mask_volume = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        slices = [(img_volume[i], mask_volume[i]) for i in range(min(len(img_volume), len(mask_volume)))]
        patient_data.append(slices)
    return train_test_split(patient_data, test_size=0.2, random_state=42)

# 4. Build U-Net Model for Binary Segmentation
def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, 3, activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.UpSampling2D()(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = layers.Conv2D(256, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, 3, activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D()(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = layers.Conv2D(128, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, 3, activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D()(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = layers.Conv2D(64, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, 3, activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D()(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = layers.Conv2D(32, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, 3, activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)  # Single output for binary segmentation

    model = models.Model(inputs, outputs)
    return model

# 5. Compile and train the model
input_shape = (256, 256, 1)  # Adjust according to your input size
unet_model = build_unet(input_shape=input_shape)
print(unet_model.summary())

unet_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=dice_loss,  # Dice Loss for binary segmentation
    metrics=[dice_coef, 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("data generation .... ")
# 6. Load data and set up the training
images_dir = 'input/images'
labels_dir = 'input/picai_labels/csPCa_lesion_delineations/human_expert/resampled'
train_data, val_data = load_patientwise_data(images_dir, labels_dir)
train_gen = ProstateDataset(train_data, batch_size=4)
val_gen = ProstateDataset(val_data, batch_size=4)
print(f"Training on {len(train_data)} patients, validating on {len(val_data)} patients.")

# 7. Calculate class weights
def calculate_class_weights(Y_train):
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(Y_train),
        y=Y_train.flatten()
    )
    return dict(enumerate(class_weights))

# Get class weights for training
class_weights = calculate_class_weights(Y_train)
print(f"Class weights: {class_weights}")

# 8. Train the model
history = unet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    batch_size=4,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10),
    ]
)

# 9. Evaluate the model
val_loss, val_accuracy, val_dice, val_precision, val_recall = unet_model.evaluate(val_gen)
print(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}, Dice Coeff: {val_dice}, Precision: {val_precision}, Recall: {val_recall}")
