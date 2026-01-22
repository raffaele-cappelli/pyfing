import math
import itertools
import numpy as np
from typing import NamedTuple
from scipy.optimize import linear_sum_assignment
from ..definitions import Minutia

def _angle_wrapping_difference(a, b):
    return abs((a - b + math.pi) % math.tau - math.pi)


def compare_minutiae_to_gt(minutiae: list[Minutia], gt_minutiae: list[Minutia], type_agnostic = True, max_distance = 16, max_direction_difference = math.pi/6) -> tuple[int, int, list]:
    _max_cost = 1e10
    max_sq_distance = max_distance**2
    C = np.full((len(gt_minutiae), len(minutiae)), _max_cost, np.float32)
    for i, (x1, y1, d1, t1, *_) in enumerate(gt_minutiae):
        for j, (x, y, d, t, *_) in enumerate(minutiae):
            if type_agnostic or t == t1:            
                dq = (x1-x)**2 + (y1-y)**2
                if dq <= max_sq_distance and _angle_wrapping_difference(d, d1) <= max_direction_difference:
                    C[i,j] = dq
    a, b = linear_sum_assignment(C)
    pairs = [(b[i], a[i]) for i in range(len(b)) if C[a[i],b[i]]!=_max_cost]
    return len(pairs), len(minutiae)-len(pairs), pairs


class MinutiaeExtractionAccuracy(NamedTuple):
    tp: float
    fp: float
    fn: float
    precision: float
    recall: float
    f1_score: float
    quality_threshold: float


def compute_minutiae_extraction_accuracy(minutiae: list[list[Minutia]], gt_minutiae: list[list[Minutia]], type_agnostic = True, max_distance = 16, 
                                         max_direction_difference = math.pi/6) -> MinutiaeExtractionAccuracy:
    all_quality_scores = [m.quality for m in itertools.chain.from_iterable(minutiae)]
    q_min, q_max = (min(all_quality_scores), max(all_quality_scores)) if len(all_quality_scores) > 0 else (0, 0)
    if q_min == q_max:
        q_max += 1e-6 # We need to perform just an iteration
    q_step = (q_max - q_min) / 100
    best = MinutiaeExtractionAccuracy(0, 0, 0, 0, 0, 0, q_min)
    for t in np.arange(q_min, q_max, q_step): 
        tp, fp, tot_gt = 0, 0, 0
        for m, gt_m in zip(minutiae, gt_minutiae):
            m = [x for x in m if x.quality >= t] # Only extracted minutiae with quality >= t
            n_true, n_false, _ = compare_minutiae_to_gt(m, gt_m, type_agnostic, max_distance, max_direction_difference)
            tp += n_true
            fp += n_false
            tot_gt += len(gt_m)
        if (tp + fp > 0) and (tot_gt > 0):
            fn = tot_gt - tp
            precision = tp / (tp+fp)
            recall = tp / tot_gt # that is /(tp+fn)
            f1 = 2*tp/(2*tp+fp+fn)
            if f1 > best.f1_score:
                best = MinutiaeExtractionAccuracy(tp, fp, fn, precision, recall, f1, t)
    return best
