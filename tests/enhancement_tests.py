##### Script to generate NIST SD27 enhanced images with both the enhancement algorithms #####

import pathlib
import time
import cv2 as cv
import pyfing as pf
from pyfing.utils.sd27 import load_sd27_test_db


def _get_roi(m, roi_border = 8):
    img_h, img_w = m.shape
    x1, y1, w, h = cv.boundingRect(m)
    x2, y2 = x1 + w, y1 + h
    x1, y1 = max(0, x1 - roi_border), max(0, y1 - roi_border)
    x2, y2 = min(img_w, x2 + roi_border), min(img_h, y2 + roi_border)
    return x1, y1, x2, y2

def _crop_roi(v, roi):
    x1, y1, x2, y2 = roi
    return v[y1:y2, x1:x2].copy()

def _run_enh_alg(db, folder_name, alg):
    pathlib.Path(f"../results/{folder_name}").mkdir(exist_ok=True)
    tot_oe, tot_fe, tot_en = 0, 0, 0
    for f, m, _, name in db:
        h, w = f.shape
        roi = _get_roi(m, 0)
        x1, y1, x2, y2 = roi
        f = _crop_roi(f, roi)
        m = _crop_roi(m, roi)
        start = time.time()
        o = pf.orientation_field_estimation(f, m)
        elapsed_oe = time.time() - start
        start = time.time()
        rp = pf.frequency_estimation(f, o, m)
        elapsed_fe = time.time() - start
        start = time.time()
        en = alg.run(f, m, o, rp)
        elapsed_en = time.time() - start
        en = cv.copyMakeBorder(en, y1, h-y2, x1, w-x2, cv.BORDER_CONSTANT)
        cv.imwrite(f"../results/{folder_name}/{name}.png", en)
        tot_oe += elapsed_oe
        tot_fe += elapsed_fe
        tot_en += elapsed_en
        print(f"{name}: {elapsed_oe:.4f}s {elapsed_fe:.4f}s {elapsed_en:.4f}")
    return tot_oe/len(db), tot_fe/len(db), tot_en/len(db)

## 

print("Loading NIST DB27...")
loader = lambda n: load_sd27_test_db('../datasets/NIST_DB27', '../datasets/orientations/SD27', n, include_minutiae_and_name=True)
db = loader('GOOD') + loader('BAD') + loader('UGLY')

for alg in [pf.Gbfen(), pf.Snfen()]:
    alg_name = type(alg).__name__.upper()
    print(f"Running {alg_name} on SD27...")
    
    oe, fe, en = _run_enh_alg(db, alg_name, alg)    
    print(f"Avg time: {oe:.4f}s {fe:.4f}s {en:.4f}s")


for alg in [pf.Gbfen(), pf.Snfen()]:
    alg_name = type(alg).__name__.upper()
    print(f"Running {alg_name} on SD27...")
    
    oe, fe, en = _run_enh_alg(db, alg_name, alg)    
    print(f"Avg time: {oe:.4f}s {fe:.4f}s {en:.4f}s")
