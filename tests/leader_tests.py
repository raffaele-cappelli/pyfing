##### Script to test LEADER minutiae extraction method #####

import math
import time
import cv2 as cv
import pyfing as pf
from pyfing.minutiae import Minutia
from pyfing.utils.sd27 import load_sd27_test_db
from pyfing.utils.iso_format import load_minutiae_from_iso_template_file
from pyfing.utils.minutiae_tools import compute_minutiae_extraction_accuracy


# FOLDER PATHS
db27_folder = "../datasets/NIST_SD27" # path of the NIST SD27 db (the folder is expected to contain ./DATA/GOOD/, ./DATA/BAD/, ./DATA/UGLY/ subfolders)
db27_seg_gt_folder = "../datasets/NIST_SD27_GT" # path of the NIST SD27 orientation/segmentation masks (the folder is expected to contain ./OF_manual subfolder)
fvc2002_db1_a_folder = "../datasets/fvc2002/db1_a" # path of the FVC2002 DB1-A image folder
fvc2002_db1_a_min_gt_folder = "../datasets/FM3_FVC2002DB1A" # path of the FVC2002 DB1-A minutiae ground truth folder
fvc2002_db1_a_seg_gt_folder = "../datasets/FVC_SEG_GT/fvc2002/db1_a" # path of the FVC2002 DB1-A segmentation ground truth folder

batch_size = 32 # Depending on the amount of GPU RAM available, this may have to be tuned
type_agnostic_modes = [True, False]
match_levels = [(16, math.pi/6), (12, math.pi/8), (8, math.pi/10)]

def _crop_roi(f, s, m, roi_border = 0):
    img_h, img_w = s.shape
    x1, y1, w, h = cv.boundingRect(s)
    x2, y2 = x1 + w, y1 + h
    x1, y1 = max(0, x1 - roi_border), max(0, y1 - roi_border)
    x2, y2 = min(img_w, x2 + roi_border), min(img_h, y2 + roi_border)
    # crop both f and s
    f = f[y1:y2, x1:x2].copy()
    s = s[y1:y2, x1:x2].copy()
    # adjust minutiae coordinates
    m = [Minutia(k.x - x1, k.y - y1, k.direction, k.type, k.quality) for k in m]
    return f, s, m


def _remove_minutiae_near_borders(minutiae, background_distance, border_distance = 14):
    h, w = background_distance.shape
    return [m for m in minutiae if 0<=m.x<w and 0<=m.y<h and background_distance[m.y, m.x] >= border_distance]


def test(db_name, alg, db, warmup = True):
    print(f"Testing on {db_name}...")
    alg.parameters.minutia_quality_threshold = 0.01 # To find the optimal F1-score
    fingerprints, segmentation_masks, gt_minutiae = map(list, zip(*[_crop_roi(f, s, m) for f, s, m, _ in db]))
    
    if warmup:
        print("Warming up...")
        for i in range(2):
            _ = alg.run_on_db(fingerprints, batch_size=batch_size)
    print("Extracting minutiae and measuring time...")
    start_time = time.time()
    minutiae = alg.run_on_db(fingerprints, batch_size=batch_size)
    elapsed = time.time() - start_time
    # Removes minutiae near the borders
    for i in range(len(gt_minutiae)):
        background_distance = cv.distanceTransform(cv.copyMakeBorder(segmentation_masks[i], 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]
        minutiae[i] = _remove_minutiae_near_borders(minutiae[i], background_distance)
        gt_minutiae[i] = _remove_minutiae_near_borders(gt_minutiae[i], background_distance)

    res = {}
    for type_agnostic in type_agnostic_modes:
        for match_level in match_levels:
            print(f"Measuring accuracy [{'Type-agnostic' if type_agnostic else 'Type-aware'}, ({match_level[0]}, π/{round(math.pi/match_level[1]):.0f})]...")
            res[(type_agnostic, match_level)] = compute_minutiae_extraction_accuracy(minutiae, gt_minutiae, type_agnostic, *match_level)

    print("--- Results ---")
    print(f"Average per-image execution time: {elapsed*1000/len(db):.1f}ms")
    print()
    print("F1-scores: ", end="")
    for type_agnostic in type_agnostic_modes:
        for match_level in match_levels:
            print(f"{res[type_agnostic, match_level].f1_score:.2f} ", end="")
    print()


###############################################################################
#################################### Main #####################################
###############################################################################

print("Loading model...")
alg = pf.Leader()

print("Loading FVC2002 DB1-A data...")
db = [(cv.imread(f'{fvc2002_db1_a_folder}/{i}_1.png', cv.IMREAD_GRAYSCALE), 
       cv.imread(f'{fvc2002_db1_a_seg_gt_folder}/{i}_1.fg.png', cv.IMREAD_GRAYSCALE), 
       load_minutiae_from_iso_template_file(f"{fvc2002_db1_a_min_gt_folder}/{i}_1.iso-fmr"), 
       f'{i}_1') for i in range(1,101)]
test("FVC2002 DB1-A", alg, db)

print("Loading NIST SD27 data...")
db = load_sd27_test_db(db27_folder, db27_seg_gt_folder, "GOOD", False, True) + \
     load_sd27_test_db(db27_folder, db27_seg_gt_folder, "BAD", False, True) + \
     load_sd27_test_db(db27_folder, db27_seg_gt_folder, "UGLY", False, True)
test("NIST SD27", alg, db)
