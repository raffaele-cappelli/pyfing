import os
import glob
import struct
from pathlib import Path
import numpy as np
import scipy as sp
import cv2 as cv

def _load_gt(path, image_w, image_h, border, step):
    """Load orientation ground truth.
       Note that FOE ground truth orientation/strength is defined over a grid with a given step: 
       it is not available for each pixel. This function returns a pixelwise orientation image where
       only the pixels where the orientation is available have strentgh > 0.
       Note about foreground mask: in FOE datasets there are specific files with this foreground
       information, but the gt orientation strength already contains the same information: it is zero
       for non-foreground blocks"""
    with open(path, 'rb') as f:
        header = struct.unpack('8s', f.read(8))[0].decode("utf-8")
        bx, by, sx, sy, w, h = struct.unpack('<llllll', f.read(24))
        data = f.read(w*h*2)
    if header!='DIRIMG00' or bx!=by or sx!=sy or border!=bx or step!=sx:
        raise Exception('Invalid data in orientation image file')
    all_orientations = np.array([x*np.pi/256 for x in data[::2]])
    all_strenghts = np.array([x/256.0 for x in data[1::2]])
    orientations = np.zeros((image_h, image_w), dtype=np.float32)
    strengths = np.zeros((image_h, image_w), dtype=np.float32)
    index = 0
    for y in range(border, image_h-border, step):
        for x in range(border, image_w-border, step):
            orientations[y, x] = all_orientations[index]
            strengths[y, x] = all_strenghts[index]
            index += 1
    # obtain mask of foreground pixels (that is where gt orientations are defined) from strengths
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    mask[strengths > 0] = 255    
    return orientations, mask


def _interpolate_orientations(source_orientations, source_mask, source_dpi, target_dpi):
    h, w = source_orientations.shape
    cos2t = np.cos(source_orientations*2)
    sin2t = np.sin(source_orientations*2)
    points = np.argwhere(source_mask)
    cos2t_values = cos2t[points[:,0], points[:,1]]
    sin2_tvalues = sin2t[points[:,0], points[:,1]]
    if source_dpi != target_dpi: # Performs resize while interpolating
        f = target_dpi / source_dpi
        points = points.astype(np.float32) * f
        h, w = int(round(h*f)), int(round(w*f))
    grid_h, grid_w = np.mgrid[:h, :w]
    i_cos2t = sp.interpolate.griddata(points, cos2t_values, (grid_h, grid_w), method='cubic')
    i_sin2t = sp.interpolate.griddata(points, sin2_tvalues, (grid_h, grid_w), method='cubic')

    # Normalization to ensure cos**2 + sin**2 = 1
    n = np.sqrt(i_cos2t**2 + i_sin2t**2)
    i_cos2t /= n
    i_sin2t /=n

    # Remove nan
    mask = np.where(np.isnan(i_cos2t) | np.isnan(i_sin2t),0,255).astype(np.uint8)
    i_cos2t[mask==0] = 0
    i_sin2t[mask==0] = 0
    
    return i_cos2t, i_sin2t, mask


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


def load_foe_train_dataset(path, target_size, target_dpi, border = 14, step = 8, folders = ['Good', 'Bad'], indices_per_folder = None):
    paths = []
    for folder_index, folder in enumerate(folders):
        p = glob.glob(path+f'{folder}/*.bmp')
        p.sort()
        if indices_per_folder is not None:
            p = [p[i] for i in indices_per_folder[folder_index]]
        paths += p
    count = len(paths)
    x = np.empty((count, target_size[1], target_size[0], 2), dtype = np.uint8)
    y = np.empty((count, target_size[1], target_size[0], 3), dtype = np.float32)
    for index in range(count):
        img_path = paths[index]
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception(f"Cannot load {img_path}")
        gt_path = Path(img_path).with_suffix('.gt')
        h, w = img.shape
        dpi = 569 if h == 560 else 500 # FVC2000/3000 images detected from image height: resolution is 569 dpi
        sparse_orientations, sparse_mask = _load_gt(gt_path, w, h, border, step)
        cos2t, sin2t, mask = _interpolate_orientations(sparse_orientations, sparse_mask, dpi, target_dpi)

        if dpi != target_dpi:
            f = target_dpi / dpi
            img = cv.resize(img, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)
        img = _adjust_size(img, target_size, img[0,0].tolist())
        cos2t = _adjust_size(cos2t, target_size, 0)
        sin2t = _adjust_size(sin2t, target_size, 0)
        mask = _adjust_size(mask, target_size, 0)

        mask01 = mask//255
        x[index] = np.dstack((img, mask01))
        y[index] = np.dstack((cos2t, sin2t, mask01))

    return x, y


def _create_pixelwise_foreground(fg, s):
    """Creates a segmentation mask starting from ground truth foreground values"""
    return cv.morphologyEx(fg, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (s, s)))


def load_foe_test_db(path):
    db = []
    with open(path + '/index.txt', 'r') as f:
        lines = f.readlines()
        n = int(lines[0])
        for i in range(n):
            t = lines[i+1].split()
            step, border = int(t[1]), int(t[2])
            fp = f'{path}/{t[0]}'            
            image = cv.imread(fp, cv.IMREAD_GRAYSCALE)
            gt_orientations, gt_mask = _load_gt(os.path.splitext(fp)[0]+'.gt', *image.shape[::-1], border, step)
            dpi = 569 if image.shape[0] == 560 else 500 # FX2000/3000 images detected from image height: resolution is 569 dpi)
            db.append( (image, _create_pixelwise_foreground(gt_mask, step), gt_orientations, gt_mask, dpi) )
    return db
