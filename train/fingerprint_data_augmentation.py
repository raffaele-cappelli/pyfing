import math
import random
import numpy as np
import cv2 as cv
import tensorflow as tf

# Data augmentation parameters
MAX_TRASLATION = 0.03
MAX_ROTATION = 20
MAX_ZOOM = 0.2
PROB_GAMMA = 0.15
MAX_GAMMA = 3
PROB_CONTRAST = 0.15

PROB_STRONG_NOISE = 0.2
STRONG_NOISE_SCRATCHES = True
STRONG_NOISE_BLOB = True
STRONG_NOISE_MORPH = True

BACK_PERCENTILE = 95
SCRATCHES_ANGLES_RANGE = (2, 6)
SCRATCHES_THICKNESS_RANGE = (2, 5)
SCRATCHES_COUNT_FACTOR = 10
SCRATCHES_BLUR_SIZE = 5
SCRATCHES_COLOR_PROB = (0.0, 0.0, 1.0) # black|white|back_percentile|noise
BLOB_COLOR_PROB = (0.0, 0.0, 1.0) # black|white|back_percentile|noise
CONTRAST_TARGET_LOW_RANGE = (100, 200)
CONTRAST_TARGET_DELTA_RANGE = (30, 60)
BLOB_MAX_COUNT = 1
BLOB_SIZE_FACTORS = (20, 10)
BLOB_MAX_ANGLE = 30


def np_augment(x, y):
    res_x, res_y = tf.numpy_function(_augmentation, [x, y], [tf.uint8, tf.uint8])
    # It seems tensors' shapes are lost after calling the numpy function: we set it again
    return tf.reshape(res_x, x.shape), tf.reshape(res_y, y.shape)


def tf_flip(x, y):
    x = tf.image.flip_left_right(x)
    x = tf.stack((x[...,0], x[...,1], (255 - x[...,2]) % 255, x[...,3]), -1) # flip orientations (in discretized [0..254] radians) too
    y = tf.image.flip_left_right(y)
    return x, y


_rnd = random.Random()
_morph_se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


def _random_affine_trasform(image, mask, orientations, rp, en, rnd):
    image_h, image_w = image.shape[0], image.shape[1]
    tx = rnd.uniform(-MAX_TRASLATION, MAX_TRASLATION) * image_w
    ty = rnd.uniform(-MAX_TRASLATION, MAX_TRASLATION) * image_h
    angle = rnd.uniform(-MAX_ROTATION, MAX_ROTATION)  # in degrees
    scale = rnd.uniform(1-MAX_ZOOM, 1+MAX_ZOOM)
    M = cv.getRotationMatrix2D((image_w/2, image_h/2), angle, scale)
    dest_size = (image_w, image_h)
    M[0, 2] += tx
    M[1, 2] += ty
    image = cv.warpAffine(image, M, dest_size, flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=image[0,0].tolist())
    mask = cv.warpAffine(mask, M, dest_size, flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)    

    # Orientations, with angle added
    angle_rad = angle * math.pi / 180
    orientations = orientations / 255 * np.pi + angle_rad
    sin2t = cv.warpAffine(np.sin(orientations*2), M, dest_size, flags=cv.INTER_CUBIC)
    cos2t = cv.warpAffine(np.cos(orientations*2), M, dest_size, flags=cv.INTER_CUBIC)    
    orientations = np.round(((np.arctan2(sin2t, cos2t) / 2) % np.pi) * 255 / np.pi).clip(0,255).astype(np.uint8)

    # Ridge-line period, with scale change
    rp = np.round(cv.warpAffine(rp, M, dest_size, flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=rp[0,0].tolist()) * scale).clip(0,255).astype(np.uint8)

    # Enhanced image
    en = cv.warpAffine(en, M, dest_size, flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT, borderValue=0)

    return image, mask, orientations, rp, en


def _gray_levels_adjustment(image, mask, rnd):
    if rnd.random() < PROB_GAMMA:
        gamma = rnd.uniform(1, MAX_GAMMA)**rnd.choice([-1, 1])
        lut = np.round(((np.arange(256)/255.0) ** (1.0/gamma)) * 255).astype(np.uint8)
        image = cv.LUT(image, lut)
   
    if rnd.random() < PROB_CONTRAST:
        g1, g2 = np.percentile(image, 5), np.percentile(image, 95)
        if g2 > g1:
            target1 = np.random.uniform(*CONTRAST_TARGET_LOW_RANGE)
            target2 = target1 + np.random.uniform(*CONTRAST_TARGET_DELTA_RANGE)
            lut = np.clip((np.arange(256) - g1) / (g2-g1) * (target2 - target1) + target1, 0, 255).astype(np.uint8)
            image = cv.LUT(image, lut)

    return image


def _random_strong_noise(image, mask, rnd):
    noise_funcs = []
    if STRONG_NOISE_SCRATCHES:
        noise_funcs.append(_random_scratches)
    if STRONG_NOISE_BLOB:
        noise_funcs.append(_random_blob)
    if STRONG_NOISE_MORPH:
        noise_funcs.append(_random_morph)
    prob = PROB_STRONG_NOISE
    while prob > 0 and noise_funcs:
        if rnd.random() < prob:
            f = rnd.choice(noise_funcs)
            image = f(image, mask, rnd)
            noise_funcs.remove(f) # To avoid applying the same noise function twice
        prob /= 2 # next one with half probability
    return image


def _random_morph(image, mask, rnd):
    op = rnd.choice([cv.MORPH_ERODE, cv.MORPH_DILATE])
    return cv.morphologyEx(image, op, _morph_se)


def _random_blob(image, mask, rnd):
    h, w = image.shape
    blob = np.zeros_like(image)
    n = rnd.randint(1, BLOB_MAX_COUNT)
    f1, f2 = BLOB_SIZE_FACTORS
    for _ in range(n):
        y, x = np.random.normal((h/2, w/2), (h/6, w/8)).astype(np.int32)
        dy, dx = np.random.uniform((h//f1, w//f1), (h//f2, w//f2)).astype(np.int32)
        angle = np.random.uniform(-BLOB_MAX_ANGLE, BLOB_MAX_ANGLE)
        cv.ellipse(blob, (x, y), (dx, dy), angle, 0, 360, 255, -1, cv.LINE_4)
    blob_mask = blob != 0
    blob_mask[mask == 0] = 0
    image[blob_mask] = np.percentile(image, BACK_PERCENTILE)
    image[blob_mask] = cv.blur(image, (15, 15))[blob_mask]
    return image


def _random_scratches(image, mask, rnd):
    h, w = image.shape
    lines = np.zeros_like(image)
    angles = np.random.choice(360, np.random.choice(range(*SCRATCHES_ANGLES_RANGE)))
    t = np.random.randint(*SCRATCHES_THICKNESS_RANGE)
    for _ in range(SCRATCHES_COUNT_FACTOR*(SCRATCHES_THICKNESS_RANGE[1]-t)):  # the less thick, the more lines
        y, x = np.random.normal((h/2, w/2), (h/6, w/8)).astype(np.int32)
        dy, dx = np.random.uniform((5, 20), (10, 200)).astype(np.int32)
        angle = np.random.choice(angles)
        cv.ellipse(lines, (x, y), (dx, dy), angle, 0, 160, 255, t, cv.LINE_AA)
    lines[mask == 0] = 0
    lines_mask = lines != 0
    p = rnd.random()
    if p < SCRATCHES_COLOR_PROB[0]:
        color = 0 
    elif p < SCRATCHES_COLOR_PROB[1]:
        color = 255
    else:
        color = np.percentile(image, BACK_PERCENTILE)
    image[lines_mask] = color
    return image


def _augmentation(x, y):
    rnd = _rnd
    image = x[..., 0] # encoding: gray scale values [0,255]
    mask = x[..., 1] # encoding: binary values [0,1]
    orientations = x[..., 2] # encoding: radians discretized in [0..255]
    rp = x[..., 3] # encoding: 10/f
    en = y[..., 0] # encoding: values in [0,255]

    image, mask, orientations, rp, en = _random_affine_trasform(image, mask, orientations, rp, en, rnd)
    image = _random_strong_noise(image, mask, rnd)
    image = _gray_levels_adjustment(image, mask, rnd)

    x = np.dstack((image, mask, orientations, rp))
    y = np.dstack((en, mask)) # the mask is also in the ground truth to be used in the loss
    return x, y
