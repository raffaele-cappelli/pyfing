##### Script to train the SUFS network on FVC datasets B #####

import os
import numpy as np
import cv2 as cv
import tensorflow as tf

from pyfing.utils.fvc_segmentation import fvc_db_non_500_dpi
from training_callbacks import ReduceLROnPlateauWithBestWeights


PATH_FVC = '../datasets/'
PATH_GT = '../datasets/segmentationbenchmark/groundtruth/'
PATH_RES = '../results/'


# Network architecture parameters
INPUT_IMAGE_SIZE = (512, 512)
INPUT_IMAGE_DPI = 500
FILTERS = [16, 32, 64, 128, 256, 512]

# Training parameters
BATCH_SIZE = 16
BATCHES_PER_EPOCH = 500
MIN_EPOCHS = 1
MAX_EPOCHS = 90
PATIENCE = 11
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.00001
LEARNING_RATE_PATIENCE = 4
LEARNING_RATE_REDUCE_FACTOR = 0.2
TVERSKY_ALPHA = 0.7

# Data augmentation parameters
AUGMENT_MAX_TRASLATION = 0.02
AUGMENT_MAX_ROTATION = 0.02
AUGMENT_MAX_CONTRAST = 0.3
AUGMENT_MAX_ZOOM = 0.2


class FvcImageAugmentation(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        b = 0
        l = tf.keras.layers    
        self.rotate_images = l.RandomRotation(AUGMENT_MAX_ROTATION, seed=seed, fill_mode="constant", fill_value=b)
        self.rotate_masks = l.RandomRotation(AUGMENT_MAX_ROTATION, seed=seed, interpolation="nearest", fill_mode="constant", fill_value=0)
        self.translate_images = l.RandomTranslation(AUGMENT_MAX_TRASLATION, AUGMENT_MAX_TRASLATION, seed=seed, fill_mode="constant", fill_value=b)
        self.translate_masks = l.RandomTranslation(AUGMENT_MAX_TRASLATION, AUGMENT_MAX_TRASLATION, seed=seed, interpolation="nearest", fill_mode="constant", fill_value=0)
        self.flip_images = l.RandomFlip("horizontal", seed=seed)
        self.flip_masks = l.RandomFlip("horizontal", seed=seed)
        self.contrast_images = l.RandomContrast((AUGMENT_MAX_CONTRAST, 0), seed=seed)
        self.zoom_images = l.RandomZoom(AUGMENT_MAX_ZOOM, seed=seed, fill_mode="constant", fill_value=b)
        self.zoom_masks = l.RandomZoom(AUGMENT_MAX_ZOOM, seed=seed, interpolation="nearest", fill_mode="constant", fill_value=0)

    def call(self, images, masks):
        masks = tf.cast(masks, tf.dtypes.float32)
        t = tf.random.uniform([], minval=1, maxval=6, dtype=tf.int32)
        if t == 1:
            images = self.rotate_images(images)
            masks = self.rotate_masks(masks)
        elif t == 2:
            images = self.translate_images(images)
            masks = self.translate_masks(masks)
        elif t == 3:
            images = self.flip_images(images)
            masks = self.flip_masks(masks)
        elif t == 4:
            images = self.contrast_images(images)
        else:
            images = self.zoom_images(images)
            masks = self.zoom_masks(masks)            
        masks = tf.cast(masks, tf.dtypes.uint8)
        return images, masks


def _adjust_size(image, target_size, border_value):
    # For each side computes crop size (if negative) or border to be added (if positive)
    h, w = image.shape
    target_w, target_h = target_size
    left = (target_w - w) // 2
    right = target_w - w - left
    top = (target_h - h) // 2
    bottom = target_h - h - top
    if left < 0 or right < 0: # Horizontal crop
        image = image[:, -left:(right if right < 0 else w)]
    if top < 0 or bottom < 0: # Vertical crop
        image = image[-top:(bottom if bottom < 0 else h)]    
    if left > 0 or right > 0 or top > 0 or bottom > 0: # Add borders
        image = cv.copyMakeBorder(image, max(0,top), max(0,bottom), max(0,left), max(0,right), cv.BORDER_CONSTANT, value = border_value)
    return image    


def _adjust_input(image, dpi, is_mask = False):
    if dpi != INPUT_IMAGE_DPI: # Resize to make its resolution INPUT_IMAGE_DPI dpi
        f = INPUT_IMAGE_DPI / dpi
        image = cv.resize(image, None, fx = f, fy = f, interpolation = cv.INTER_NEAREST if is_mask else cv.INTER_CUBIC)
    return _adjust_size(image, INPUT_IMAGE_SIZE, 0 if is_mask else image[0,0].tolist())


def load_fvc_dataset(db_years, db_numbers, db_set, finger_from, finger_to, impression_from, impression_to):
    i1, i2 = finger_from, finger_to
    j1, j2 = impression_from, impression_to

    count = len(db_years) * len(db_numbers) * (i2-i1+1) * (j2-j1+1)
    x = np.empty((count, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0], 1), dtype = np.float32)
    y = np.empty((count, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0], 1), dtype = np.uint8)
    index = 0

    for year in db_years:
        for n in db_numbers:
            dpi = fvc_db_non_500_dpi.get((year, n), 500)
            for i in range(i1, i2+1):
                for j in range(j1, j2+1):
                    img_path = f'{PATH_FVC}fvc{year}/db{n}_{db_set}/{i}_{j}.png'
                    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                    if img is None:
                        raise Exception(f"Cannot load {img_path}")
                    img = _adjust_input(img, dpi)
                    gt_path = f'{PATH_GT}/fvc{year}_db{n}_im_{i}_{j}seg.png'
                    gt = cv.imread(gt_path, cv.IMREAD_GRAYSCALE)
                    if gt is None:
                        raise Exception(f"Cannot load {gt_path}")
                    gt = 255 - gt # FVC ground truth is 0 for foreground and 255 for background: we want 255 for foreground and 0 for background
                    gt = _adjust_input(gt, dpi, True) // 255 # 0,255 -> 0,1
                    x[index] = img[..., np.newaxis]
                    y[index] = gt[..., np.newaxis]                        
                    index += 1
    return tf.data.Dataset.from_tensor_slices((x, y))


def build_sufs_model():
    layers = tf.keras.layers
    img_size = INPUT_IMAGE_SIZE[::-1] # (w, h) ==> (h, w) = (rows, cols)
    inputs = layers.Input(img_size + (1,))
    x = inputs    
    level_outputs = []
    for filters in FILTERS:
        x = layers.Conv2D(filters, 3, padding="same", activation = "relu")(x)
        x = layers.BatchNormalization()(x)
        level_outputs.append(x)
        x = layers.MaxPooling2D(2, padding="same")(x)
    for filters, lo in zip(reversed(FILTERS), reversed(level_outputs)):        
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, lo])
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
    return tf.keras.Model(inputs, outputs)


def tversky_index(y_true, y_pred):
    smooth = 1.
    y_true_pos = tf.keras.backend.cast(tf.keras.backend.flatten(y_true), tf.dtypes.float32)
    y_pred_pos = tf.keras.backend.cast(tf.keras.backend.flatten(y_pred), tf.dtypes.float32)
    true_pos = tf.keras.backend.sum(y_true_pos * y_pred_pos)
    false_neg = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)    
    return (true_pos + smooth) / (true_pos + TVERSKY_ALPHA * false_neg + (1 - TVERSKY_ALPHA) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky_index(y_true, y_pred)


def main():
    print("Loading data...")
    train_ds = load_fvc_dataset([2000,2002,2004], [1,2,3,4], "b", 102, 110, 1, 8)
    train_ds = train_ds.cache().shuffle(1000).repeat().batch(BATCH_SIZE).map(FvcImageAugmentation()).prefetch(tf.data.AUTOTUNE)
    val_ds = load_fvc_dataset([2000,2002,2004], [1,2,3,4], "b", 101, 101, 1, 8)
    val_ds = val_ds.batch(BATCH_SIZE)
    log_path = f"{PATH_RES}training-SUFS"
    os.makedirs(log_path)
    print("Preparing model...")
    model = build_sufs_model()
    model.summary()    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=tversky_loss)
    
    print("Training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, start_from_epoch=MIN_EPOCHS, min_delta=0.00003, verbose = 1),
        ReduceLROnPlateauWithBestWeights(factor=LEARNING_RATE_REDUCE_FACTOR, patience=LEARNING_RATE_PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1, min_delta=0.00003),
        tf.keras.callbacks.CSVLogger(f'{log_path}/training.csv', append=True)
    ]

    model.fit(train_ds, epochs=MAX_EPOCHS, steps_per_epoch=BATCHES_PER_EPOCH, validation_data=val_ds, callbacks=callbacks)
    print("Saving model...")
    model.save(f'{log_path}/model.h5', include_optimizer=False)


##

main()
