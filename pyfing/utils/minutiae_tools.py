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


def compute_minutiae_extraction_accuracy(
    minutiae: list[list[Minutia]], 
    gt_minutiae: list[list[Minutia]], 
    type_agnostic: bool = True, 
    max_distance: int = 16, 
    max_direction_difference: float = math.pi / 6
) -> MinutiaeExtractionAccuracy:
    """
    Computes the optimal minutiae extraction accuracy by dynamically finding
    the quality threshold that maximizes the F1-score.
    
    Optimized with an O(1) list-based cache and adaptive percentile thresholds 
    to handle continuous quality values efficiently.
    """
    # 1. Extract all unique quality values from the entire dataset using an efficient set comprehension
    unique_qualities = sorted({m.quality for m in itertools.chain.from_iterable(minutiae)})
    
    if not unique_qualities:
        return MinutiaeExtractionAccuracy(0, 0, 0, 0, 0, 0, 0)
        
    # 2. Select up to 100 thresholds.
    # If we have <= 100 unique values, we use them all directly (perfect resolution, zero overhead).
    # If we have more, we sample 100 values at equal percentile ranks of the distribution.
    if len(unique_qualities) <= 100:
        thresholds = unique_qualities
    else:
        # Exact nearest-neighbor percentile selection in pure Python
        n_elements = len(unique_qualities)
        thresholds = [unique_qualities[round(i * (n_elements - 1) / 99)] for i in range(100)]
        
    best = MinutiaeExtractionAccuracy(0, 0, 0, 0, 0, 0, thresholds[0])
    
    # Pre-allocate a list to cache the single most recent state for each image.
    # Since active minutiae counts strictly decrease or stay the same as threshold t increases,
    # we only ever need to remember the immediate previous step.
    # Format: [None] or [(last_active_count, cached_tp, cached_fp)]
    matching_cache: list[tuple[int, int, int] | None] = [None] * len(minutiae)
    
    # Pre-calculate the total number of ground truth minutiae (invariant)
    tot_gt = sum(len(gt_m) for gt_m in gt_minutiae)
    if tot_gt == 0:
         return best

    # 3. Iterate through adaptive quality thresholds
    for t in thresholds: 
        tp, fp = 0, 0
        
        for idx, (m, gt_m) in enumerate(zip(minutiae, gt_minutiae)):
            # Filter minutiae above the current quality threshold
            filtered_m = [x for x in m if x.quality >= t]
            num_active = len(filtered_m)
            
            # Cache hit check: since filtered_m is a deterministic subset,
            # if the count of active minutiae hasn't changed compared to the previous step,
            # we can skip the geometric matching entirely.
            cached_state = matching_cache[idx]
            if cached_state is not None and cached_state[0] == num_active:
                tp += cached_state[1]
                fp += cached_state[2]
                continue
            
            # Cache miss: perform the exact bipartite geometric matching
            n_true, n_false, _ = compare_minutiae_to_gt(
                filtered_m, gt_m, type_agnostic, max_distance, max_direction_difference
            )
            
            # Overwrite the cache with the new active state for this image
            matching_cache[idx] = (num_active, n_true, n_false)
            
            tp += n_true
            fp += n_false
            
        # 4. Calculate metrics and update the best score
        if (tp + fp > 0):
            fn = tot_gt - tp
            precision = tp / (tp + fp)
            recall = tp / tot_gt
            f1 = 2 * tp / (2 * tp + fp + fn)
            
            if f1 > best.f1_score:
                best = MinutiaeExtractionAccuracy(tp, fp, fn, precision, recall, f1, t)
                
    return best


