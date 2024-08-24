import numpy as np
import cv2 as cv
from ..frequencies import compute_ridge_period_MAPE


def load_ffe_dataset(base_path, folder):
    indices = range(50) if folder == "Bad" else range(110,120)
    db = []
    for i in indices:
        path = f"{base_path}/{folder}/{i:02d}"
        f = cv.imread(path + ".png", cv.IMREAD_GRAYSCALE)
        m = cv.imread(path + ".fg.png", cv.IMREAD_GRAYSCALE)
        discretized_o = cv.imread(path + ".or.png", cv.IMREAD_GRAYSCALE)
        o = discretized_o.astype(np.float32) / 255.0 * np.pi
        rp = cv.imread(path + ".fr.png", cv.IMREAD_GRAYSCALE).astype(np.float32) / 10
        db.append((f, m, o, rp))
    return db


def compute_average_error_on_db(alg, db, stft_downsampling = False):
    s = 0
    for f, m, o, gt_rp in db:
        rp = alg.run(f, m, o)
        if stft_downsampling:
            rp, gt_rp, m = _downsample_as_stft(rp), _downsample_as_stft(gt_rp), _downsample_as_stft(m)
        s += compute_ridge_period_MAPE(rp, gt_rp, m)
    return s / len(db)


def _downsample_as_stft(x):
    return x[22::12, 22::12].copy()

