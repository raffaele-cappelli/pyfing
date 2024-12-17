##### Script to test both orientation estimation algorithms on NIST SD27 and FOE-TEST benchmarks #####

import numpy as np
import cv2 as cv
import pyfing as pf
from pyfing.orientations import compute_orientation_RMSD
from pyfing.utils.sd27 import load_sd27_test_db
from pyfing.utils.foe import load_foe_test_db

def _compute_average_error_on_db(db, alg, roi_border = 8):
    sum_err = 0
    for f, m, gt_or, gt_mask, dpi in db:
        img_h, img_w = f.shape
        x1, y1, w, h = cv.boundingRect(m)
        x2, y2 = x1 + w, y1 + h
        x1, y1 = max(0, x1 - roi_border), max(0, y1 - roi_border)
        x2, y2 = min(img_w, x2 + roi_border), min(img_h, y2 + roi_border)
        w, h = x2 - x1, y2 - y1
        roi_img = f[y1:y1+h,x1:x1+w]
        roi_mask = m[y1:y1+h,x1:x1+w]
        roi_or, _ = alg.run(roi_img, roi_mask, dpi)
        orientations = np.zeros(f.shape, np.float32)
        orientations[y1:y1+h,x1:x1+w] = roi_or
        sum_err += compute_orientation_RMSD(orientations, gt_or, gt_mask)
    return sum_err / len(db)

## 

for alg in [pf.Gbfoe(), pf.Snfoe()]:
    loader_foe = lambda n: load_foe_test_db('../datasets/FOE-Test/' + n)
    loader_sd27 = lambda n: load_sd27_test_db('../datasets/NIST_DB27', '../datasets/orientations/SD27', n, include_orientations_and_dpi=True)
    print("Loading FOE-TEST datasets...")
    _db_foe_good, _db_foe_bad = loader_foe('Good'), loader_foe('Bad')
    print("Loading SD27 datasets...")
    _db_good, _db_bad, _db_ugly = loader_sd27('GOOD'), loader_sd27('BAD'), loader_sd27('UGLY')
    print("Testing on FOE-TEST...")
    eg_foe = _compute_average_error_on_db(_db_foe_good, alg)
    eb_foe = _compute_average_error_on_db(_db_foe_bad, alg)
    print(f"Tested {type(alg).__name__} on FOE-TEST. RMSD: {eg_foe:.2f}° / {eb_foe:.2f}°")
    print("Testing on SD27...")
    eg = _compute_average_error_on_db(_db_good, alg)
    eb = _compute_average_error_on_db(_db_bad, alg)
    eu = _compute_average_error_on_db(_db_ugly, alg)
    ea = (eg*len(_db_good)+eb*len(_db_bad)+eu*len(_db_ugly)) / (len(_db_good)+len(_db_bad)+len(_db_ugly))
    print(f"Tested {type(alg).__name__} on NIST SD27. RMSD: {ea: .2f}° | {eg:.2f}° / {eb:.2f}° / {eu:.2f}°")

