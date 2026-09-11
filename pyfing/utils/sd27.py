import math
from glob import glob
import numpy as np
import cv2 as cv
from pyfing.definitions import Minutia
from ._nist_format import _read_nist_file


def _parse_minutia(xyd, q, t) -> Minutia:
    x = int(round(float(xyd[0:4])*19.69/100))
    y = int(round((3900-float(xyd[4:8]))*19.69/100))
    d = (math.radians(int(xyd[8:11])) + math.pi) % (2*math.pi)
    q = int(q)
    t = 'E' if t == 'A' else 'B' if t == 'B' else 'O'
    return Minutia(x, y, d, t, q)


def _load_sd27_orientations(path, border):
    shape = (48*16, 50*16)
    values = np.loadtxt(path, np.float32, delimiter=',')
    orientations = np.zeros(shape, np.float32)
    mask = np.zeros(shape, np.uint8)
    for i in range(48):
        for j in range(50):
            m = 255 if values[i, j]!=91 else 0
            mask[i*16+border, j*16+border] = m
            if m != 0:
                orientations[i*16+border, j*16+border] = values[i, j] * np.pi / 180  
    return mask, orientations    


def _create_pixelwise_foreground(fg, s):
    """Creates a segmentation mask starting from ground truth foreground values"""
    return cv.morphologyEx(fg, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (s, s)))


def load_sd27_test_db(image_path, gt_path, db_name, include_orientations_and_dpi = False, include_minutiae_and_name = False):
    """
    Loads the latent SD27 dataset with the specified name ("GOOD", "BAD", or "UGLY") from the given path.
    The database is returned as a list of tuples, each containing:
    - The fingerprint image (numpy array)
    - The ground truth pixelwise segmentation mask (numpy array)
    - Optionally, the ground truth blockwise orientation field and foreground mask (numpy arrays), and the DPI of the image (int)
    - Optionally, the ground truth minutiae (list of Minutia objects) and the name of the image (str)    
    """

    img_paths = sorted(glob(f"{image_path}/DATA/{db_name}/**/*L*.EFT", recursive=True))
    border_gt = 8
    db = []
    for index in range(len(img_paths)):
        fields = _read_nist_file(img_paths[index])
        w_input = int(fields["13.006"][0][0])
        h_input = int(fields["13.007"][0][0])
        img = np.frombuffer(fields["13.999"], np.uint8).reshape(h_input, w_input)
        img_name = img_paths[index][-11:-7]
        gt_mask, gt_orientations = _load_sd27_orientations(f"{gt_path}/OF_manual/{img_name}.txt", border_gt)
        mask = _create_pixelwise_foreground(gt_mask, 16)
        fields_min = _read_nist_file(img_paths[index][:-5]+"I.LFF")
        minutiae = [_parse_minutia(xyd, q, t) for _, xyd, q, t in fields_min['9.012']]
        t = [img, mask]
        if include_orientations_and_dpi:
            t += [gt_orientations, gt_mask, 500]
        if include_minutiae_and_name:
            t += [minutiae, img_paths[index][-11:-4]]
        db.append(tuple(t))
    return db
