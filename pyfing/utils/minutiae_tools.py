import math
import itertools
import numpy as np
from typing import NamedTuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
import concurrent.futures
from ..definitions import Minutia

def _angle_wrapping_difference(a, b):
    return abs((a - b + math.pi) % math.tau - math.pi)


def compare_minutiae(minutiae1: list[Minutia], minutiae2: list[Minutia], type_agnostic = True, max_distance = 16, max_direction_difference = math.pi/6) -> tuple[int, int, list]:    
    """
    Compares two sets of minutiae and returns the number of true positives, false positives, and the list of matched pairs.
    """
    _max_cost = 1e10
    max_sq_distance = max_distance**2
    C = np.full((len(minutiae2), len(minutiae1)), _max_cost, np.float32)
    for i, (x1, y1, d1, t1, *_) in enumerate(minutiae2):
        for j, (x, y, d, t, *_) in enumerate(minutiae1):
            if type_agnostic or t == t1:            
                dq = (x1-x)**2 + (y1-y)**2
                if dq <= max_sq_distance and _angle_wrapping_difference(d, d1) <= max_direction_difference:
                    C[i,j] = dq
    a, b = linear_sum_assignment(C)
    pairs = [(b[i], a[i]) for i in range(len(b)) if C[a[i],b[i]]!=_max_cost]
    return len(pairs), len(minutiae1)-len(pairs), pairs


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
            n_true, n_false, _ = compare_minutiae(
                filtered_m, gt_m, type_agnostic, max_distance, max_direction_difference
            )
            
            # Overwrite the cache with the new active state for this image
            matching_cache[idx] = (num_active, n_true, n_false)
            
            tp += n_true
            fp += n_false
            
        # 4. Calculate metrics and update the best score
        fn = tot_gt - tp
        if (tp + fp > 0):
            precision = tp / (tp + fp)
            recall = tp / tot_gt if tot_gt > 0 else 0.0 # Handle edge case: no ground truth minutiae but predictions exist, zero recall
            f1 = 2 * tp / (2 * tp + fp + fn)
        else:
            if tot_gt == 0:
                precision, recall, f1 = 1.0, 1.0, 1.0  # Edge case: no ground truth and no predictions, perfect score
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0 # Edge case: no predictions but ground truth exists, zero score
            
        if f1 > best.f1_score:
            best = MinutiaeExtractionAccuracy(tp, fp, fn, precision, recall, f1, t)
                
    return best



def _compare_minutiae_to_gt_optimized(
    m_coords: np.ndarray,
    m_dirs: np.ndarray,
    m_types: np.ndarray,
    gt_coords: np.ndarray,
    gt_dirs: np.ndarray,
    gt_types: np.ndarray,
    gt_tree: KDTree,
    type_agnostic: bool = True,
    max_distance: float = 16.0,
    max_direction_difference: float = math.pi / 6
) -> tuple[int, int]:
    """
    Performs exact Hungarian matching between the filtered minutiae subset and the ground truth.
    Uses a KD-Tree and NumPy vectorization to reduce computation time.
    """
    n_gt = gt_coords.shape[0]
    n_m = m_coords.shape[0]

    if n_m == 0 or n_gt == 0:
        return 0, n_m

    # 1. Find only neighbors within 'max_distance' using the KD-Tree built on the ground truth
    # query_ball_point returns the indices of nearby ground-truth points for each minutia
    neighbor_indices = gt_tree.query_ball_point(m_coords, r=max_distance)

    _max_cost = 1e10
    C = np.full((n_gt, n_m), _max_cost, dtype=np.float32)
    has_valid_pair = False

    # 2. Populate only cells corresponding to spatially nearby minutiae
    for j, gt_idx_list in enumerate(neighbor_indices):
        if not gt_idx_list:
            continue

        gt_idx_arr = np.array(gt_idx_list)

        # Filter by minutia type
        if not type_agnostic:
            type_mask = (gt_types[gt_idx_arr] == m_types[j])
            gt_idx_arr = gt_idx_arr[type_mask]
            if len(gt_idx_arr) == 0:
                continue

        # Vectorized angular direction filter: abs((a - b + pi) % tau - pi)
        d_diff = np.abs((gt_dirs[gt_idx_arr] - m_dirs[j] + math.pi) % math.tau - math.pi)
        valid_mask = d_diff <= max_direction_difference
        valid_gt_idx = gt_idx_arr[valid_mask]

        if len(valid_gt_idx) > 0:
            # Squared Euclidean distance
            dists_sq = np.sum((gt_coords[valid_gt_idx] - m_coords[j]) ** 2, axis=1)
            C[valid_gt_idx, j] = dists_sq
            has_valid_pair = True

    if not has_valid_pair:
        return 0, n_m

    # 3. Exact Hungarian matching on the reduced matrix
    row_ind, col_ind = linear_sum_assignment(C)
    
    # Count only matches with a valid cost (below _max_cost)
    n_true = int(np.sum(C[row_ind, col_ind] < _max_cost))
    n_false = n_m - n_true
    return n_true, n_false


def _process_single_image(args) -> tuple[int, np.ndarray, np.ndarray]:
    """Helper function run in parallel for a single fingerprint.

    Returns (n_gt, tp_per_threshold, fp_per_threshold): n_gt is fixed for
    the fingerprint, while tp/fp are arrays aligned with 'thresholds', to be
    used for the subsequent image-level bootstrap.
    """
    m_list, gt_list, thresholds, type_agnostic, max_dist, max_dir_diff = args
    n_thresholds = len(thresholds)

    n_gt, n_m = len(gt_list), len(m_list)
    if n_m == 0:
        return n_gt, np.zeros(n_thresholds, dtype=np.int32), np.zeros(n_thresholds, dtype=np.int32)

    m_qualities = np.array([m.quality for m in m_list], dtype=np.float32)

    if n_gt == 0:
        # No ground truth, all minutiae that are above the quality threshold are false positives
        fp_arr = np.array([np.count_nonzero(m_qualities >= t) for t in thresholds], dtype=np.int32)
        return 0, np.zeros(n_thresholds, dtype=np.int32), fp_arr

    # Convert the entire data to NumPy arrays
    m_coords = np.array([[m.x, m.y] for m in m_list], dtype=np.float32)
    m_dirs = np.array([m.direction for m in m_list], dtype=np.float32)
    m_types = np.array([m.type == 'E' for m in m_list], dtype=np.bool_)

    gt_coords = np.array([[g.x, g.y] for g in gt_list], dtype=np.float32)
    gt_dirs = np.array([g.direction for g in gt_list], dtype=np.float32)
    gt_types = np.array([g.type == 'E' for g in gt_list], dtype=np.bool_)

    # Build the KD-Tree on the ground truth (only ONCE per fingerprint)
    gt_tree = KDTree(gt_coords)

    tp_arr = np.zeros(n_thresholds, dtype=np.int32)
    fp_arr = np.zeros(n_thresholds, dtype=np.int32)
    last_count = -1
    last_tp, last_fp = 0, 0

    for i, t in enumerate(thresholds):
        mask = m_qualities >= t
        num_active = np.count_nonzero(mask)

        if num_active == 0:
            last_count, last_tp, last_fp = 0, 0, 0
            continue

        # Cache hit: if the number of minutiae above the threshold has not changed,
        # the filtered minutiae are identical to those from the previous step
        if num_active == last_count:
            tp_arr[i], fp_arr[i] = last_tp, last_fp
            continue

        last_tp, last_fp = _compare_minutiae_to_gt_optimized(m_coords[mask], m_dirs[mask], m_types[mask], 
                                                             gt_coords, gt_dirs, gt_types, gt_tree, type_agnostic, max_dist, max_dir_diff) # pyright: ignore[reportArgumentType]
        last_count = num_active
        tp_arr[i], fp_arr[i] = last_tp, last_fp

    return n_gt, tp_arr, fp_arr


def _edge_case_accuracy(num_gt, num_m) -> MinutiaeExtractionAccuracy:
    """Handle edge cases where there are no minutiae or no ground truth."""
    if num_gt == 0 and num_m == 0:
        return MinutiaeExtractionAccuracy(0, 0, 0, 1.0, 1.0, 1.0, 0.0)
    elif num_gt == 0:
        return MinutiaeExtractionAccuracy(0, num_m, 0, 0.0, 1.0, 0.0, 0.0)
    elif num_m == 0:
        return MinutiaeExtractionAccuracy(0, 0, num_gt, 1.0, 0.0, 0.0, 0.0)
    else:
        raise ValueError("Unexpected case: both num_gt and num_m are non-zero in _edge_case_accuracy.")


def _bootstrap_f1_ci(
    tp_per_image: np.ndarray,
    fp_per_image: np.ndarray,
    n_gt_per_image: np.ndarray,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None
) -> tuple[float, float]:
    """Compute the confidence interval (2.5th-97.5th percentile) for F1 at the optimal threshold,
    using image-level bootstrap (resampling with replacement)."""
    n_images = tp_per_image.shape[0]
    if n_images == 0:
        return 0.0, 0.0

    rng = rng if rng is not None else np.random.default_rng()
    sample_idx = rng.integers(0, n_images, size=(n_bootstrap, n_images))

    tp_sum = tp_per_image[sample_idx].sum(axis=1)
    fp_sum = fp_per_image[sample_idx].sum(axis=1)
    gt_sum = n_gt_per_image[sample_idx].sum(axis=1)
    fn_sum = gt_sum - tp_sum

    denom = 2 * tp_sum + fp_sum + fn_sum
    boot_f1 = np.divide(2 * tp_sum, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)

    f1_lo, f1_hi = np.percentile(boot_f1, [2.5, 97.5])
    return float(f1_lo), float(f1_hi)


def compute_minutiae_extraction_accuracy_ex(
    minutiae: list[list[Minutia]], 
    gt_minutiae: list[list[Minutia]], 
    type_agnostic: bool = True, 
    max_distance: int = 16, 
    max_direction_difference: float = math.pi / 6,
    max_workers: int = 4,
    n_bootstrap: int = 1000,
    bootstrap_seed: int | None = None
) -> tuple[MinutiaeExtractionAccuracy, list[tuple[float, float, float, float]], tuple[float, float]]:

    """
    Computes the optimal minutiae extraction accuracy by dynamically finding
    the quality threshold that maximizes the F1-score, using parallel processing
    and bootstrap confidence intervals for the F1-score at the optimal threshold.
    With respect to compute_minutiae_extraction_accuracy(), this function adds:
    - KDTree for efficient nearest neighbor search.
    - Various other optimizations to reduce the number of geometric comparisons.
    - Parallel processing of images for efficiency.
    - Bootstrap confidence intervals for the F1-score at the optimal threshold.

    Parameters:
    - minutiae: List of lists of detected minutiae for each image.
    - gt_minutiae: List of lists of ground truth minutiae for each image.
    - type_agnostic: If True, ignore minutiae types during matching.
    - max_distance: Maximum distance for matching minutiae.
    - max_direction_difference: Maximum direction difference for matching minutiae.
    - max_workers: Number of parallel workers for processing images.
    - n_bootstrap: Number of bootstrap samples for confidence interval estimation.
    - bootstrap_seed: Random seed for reproducibility of bootstrap sampling.
    Returns a tuple containing:
    - best: MinutiaeExtractionAccuracy object with the best metrics and threshold.
    - pr_values: List of tuples (threshold, precision, recall, f1_score)
      for each threshold evaluated.
    - f1_ci: Tuple (f1_lo, f1_hi) representing the 95% confidence interval for the F1-score at the optimal threshold.
    """

    # 1. Extract unique or sampled thresholds
    unique_qualities = sorted({m.quality for m in itertools.chain.from_iterable(minutiae)})
    tot_gt = sum(len(gt) for gt in gt_minutiae)
    if len(unique_qualities) == 0 or tot_gt == 0:
        res = _edge_case_accuracy(tot_gt, sum(len(x) for x in minutiae))
        return res, [(0.0, res.precision, res.recall, res.f1_score)], (res.f1_score, res.f1_score)

    if len(unique_qualities) <= 100:
        thresholds = unique_qualities
    else:
        n_elements = len(unique_qualities)
        thresholds = [unique_qualities[round(i * (n_elements - 1) / 99)] for i in range(100)]

    # 2. Process fingerprints in parallel
    tasks = [
        (m, gt, thresholds, type_agnostic, max_distance, max_direction_difference)
        for m, gt in zip(minutiae, gt_minutiae)
    ]

    n_images = len(tasks)
    n_thresholds = len(thresholds)
    tp_per_image = np.zeros((n_images, n_thresholds), dtype=np.int32)
    fp_per_image = np.zeros((n_images, n_thresholds), dtype=np.int32)
    n_gt_per_image = np.zeros(n_images, dtype=np.int64)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for img_idx, (img_n_gt, tp_arr, fp_arr) in enumerate(executor.map(_process_single_image, tasks, chunksize=32)):
            n_gt_per_image[img_idx] = img_n_gt
            tp_per_image[img_idx] = tp_arr
            fp_per_image[img_idx] = fp_arr

    # Global counts for each threshold (summed across all images)
    tp_per_t = tp_per_image.sum(axis=0)
    fp_per_t = fp_per_image.sum(axis=0)

    # 3. Calculate metrics and find the best F1 score
    best = MinutiaeExtractionAccuracy(0, 0, 0, 0, 0, 0, thresholds[0])
    best_idx = 0

    pr_values = []
    for i, t in enumerate(thresholds):
        tp = int(tp_per_t[i])
        fp = int(fp_per_t[i])
        fn = tot_gt - tp

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            recall = tp / tot_gt
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0
        pr_values.append((t, precision, recall, f1))

        if f1 > best.f1_score:
            best = MinutiaeExtractionAccuracy(tp, fp, fn, precision, recall, f1, float(t))
            best_idx = i

    # 4. Bootstrap confidence interval for F1 at the optimal threshold (best_idx is fixed)
    rng = np.random.default_rng(bootstrap_seed)
    f1_ci = _bootstrap_f1_ci(
        tp_per_image[:, best_idx], fp_per_image[:, best_idx], n_gt_per_image,
        n_bootstrap=n_bootstrap, rng=rng
    )

    return best, pr_values, f1_ci