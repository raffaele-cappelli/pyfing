##### Script to test SBMEX minutiae extraction method #####

import itertools
import math
import time
import numpy as np
import cv2 as cv
from tqdm import tqdm
import pyfing as pf
from pyfing.definitions import Minutia
from pyfing.utils.sd302 import load_sd302_exemplar_db 
from pyfing.utils.minutiae_tools import compute_minutiae_extraction_accuracy_ex


# --------------------
# --- FOLDER PATHS ---
# --------------------

# path of the NIST SD302 db (SD 302g: annotation records of exemplar fingerprint images from SD 302b) [https://doi.org/10.6028/NIST.TN.2367]
db302_folder = '../datasets/NIST_SD302/sd302g/ebts'

# --------------------


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


def _segmentation_gt_for_sd302_exemplar(f):
    """
    SD302 Exemplar set does not provide a segmentation mask, but all images have a perfectly-white background, 
    so we can use a simple thresholding approach to obtain a good ground-truth segmentation mask.
    """
    avg = cv.boxFilter(f, -1, (15, 15))    
    m = (avg < 240).astype(np.uint8)*255
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(m)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        largest_cc_mask = np.uint8(labels == largest_label) * 255
        contours, _ = cv.findContours(largest_cc_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        largest_contour = contours[0]
        poly = cv.convexHull(largest_contour)
        m[:] = 0
        cv.drawContours(m, [poly], -1, 255, thickness=cv.FILLED) # pyright: ignore[reportArgumentType, reportCallIssue]
    return m

def test(db_name: str, alg: pf.Sbmex, db: list[tuple]):
    print(f"Testing on {db_name} ({len(db)} fingerprints)...")
    alg.parameters.minutia_quality_threshold = 0.01 # To find the optimal F1-score
    fingerprints, gt_segmentation_masks, gt_minutiae = map(list, zip(*[_crop_roi(f, s, m) for f, s, m, _ in db]))

    segmentation_masks, enhanced_images = [], []
    enh_start_time = time.time()
    for f in tqdm(fingerprints, desc="Enhancing fingerprints with the traditional pipeline"):
        segmentation_masks.append(mask := pf.fingerprint_segmentation(f, method="GMFS"))
        orient = pf.orientation_field_estimation(f, mask, method="GBFOE")
        freq = pf.frequency_estimation(f, orient, mask, method="XSFFE")
        enhanced_images.append(pf.fingerprint_enhancement(f, orient, freq, mask, method="GBFEN"))
    enh_elapsed = time.time() - enh_start_time

    minutiae = []
    start_time = time.time()
    for e, m in tqdm(zip(enhanced_images, segmentation_masks), desc="Detecting minutiae with SBMEX", total=len(enhanced_images)):
        minutiae.append(alg.run(e, m))
    elapsed = time.time() - start_time

    # Removes minutiae near the borders
    for i in range(len(gt_minutiae)):
        background_distance = cv.distanceTransform(cv.copyMakeBorder(gt_segmentation_masks[i], 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]
        minutiae[i] = _remove_minutiae_near_borders(minutiae[i], background_distance)
        gt_minutiae[i] = _remove_minutiae_near_borders(gt_minutiae[i], background_distance)

    res = {}
    for type_agnostic, match_level in tqdm(list(itertools.product(type_agnostic_modes, match_levels)), desc="Computing accuracy", total=len(type_agnostic_modes)*len(match_levels)):
        res[(type_agnostic, match_level)] = compute_minutiae_extraction_accuracy_ex(minutiae, gt_minutiae, type_agnostic, *match_level)

    print()
    print("--- Results ---")
    print(f"Average per-image enhancement pipeline time: {enh_elapsed*1000/len(db):.1f}ms")
    print(f"Average per-image minutiae detection time: {elapsed*1000/len(db):.1f}ms")
    print()
    print("F1-scores: ")
    for type_agnostic in type_agnostic_modes:
        for match_level in match_levels:
            r = res[type_agnostic, match_level]
            f1_lo, f1_hi = r[2]
            print(f"{'Type-agnostic' if type_agnostic else 'Type-aware'} {match_level[0]}, π/{round(math.pi/match_level[1]):.0f}: {r[0].f1_score:.2f} [{f1_lo:.2f}, {f1_hi:.2f}]")
    print()


###############################################################################
#################################### Main #####################################
###############################################################################

alg = pf.Sbmex()
print("Loading NIST SD302 data...")
db = [(f, _segmentation_gt_for_sd302_exemplar(f), gt_minutiae, name) for  f, gt_minutiae, name, _ in load_sd302_exemplar_db(db302_folder)]
test("NIST SD302", alg, db)
