import math
from dataclasses import dataclass
import numpy as np
import cv2 as cv
from abc import abstractmethod, ABC
from .definitions import Minutia, Image, Parameters
from pyfing.utils.minutiae_tools import compare_minutiae
from scipy.spatial.distance import pdist, squareform


class MinutiaExtractionFromEnhancedImageParameters(Parameters):
    """
    Base class for the parameters of a minutia extraction method that operates on enhanced images and their corresponding segmentation masks.
    """
    pass


class MinutiaExtractionFromEnhancedImageAlgorithm(ABC):
    """
    Base class for minutia extraction methods that operate on enhanced images and their corresponding segmentation masks.
    """
    def __init__(self, parameters: MinutiaExtractionFromEnhancedImageParameters):
        self.parameters = parameters
    
    @abstractmethod
    def run(self, image: Image, mask: Image, dpi: int = 500, intermediate_results : list | None = None) -> list[Minutia]:
        raise NotImplementedError
    
    def run_on_db(self, images: list[Image], masks: list[Image], dpi_of_images: list[int]|None = None) -> list[list[Minutia]]:
        dpi_list = [500] * len(images) if dpi_of_images is None else dpi_of_images
        return [self.run(img, mask, dpi) for img, mask, dpi in zip(images, masks, dpi_list)]



def _compute_crossing_number(values):
    """Computes the crossing number counting 1->0 transitions in the ordered neighborhood loop."""
    return (values > np.roll(values, -1)).sum()


def _compute_next_ridge_following_directions(previous_direction, values):    
    next_positions = np.argwhere(values!=0).ravel().tolist()
    if len(next_positions) > 0 and previous_direction != 8:
        # There is a previous direction: return all the next directions, sorted according to the distance from it,
        #                                except the direction, if any, that corresponds to the previous position
        next_positions.sort(key = lambda d: 4 - abs(abs(d - previous_direction) - 4))
        if next_positions[-1] == (previous_direction + 4) % 8: # the direction of the previous position is the opposite one
            next_positions = next_positions[:-1] # removes it
    return next_positions        


def _angle_difference(a, b):
    """
    Computes the minimum absolute difference between two angles.
    Works correctly for angles wrapped within the [-pi, pi] range.
    """
    diff = abs(a - b)
    return diff if diff <= math.pi else math.tau - diff


def _angle_mean(a, b):
    """Computes the circular mean of two angles using their directional components."""
    return math.atan2((math.sin(a)+math.sin(b))/2, ((math.cos(a)+math.cos(b))/2))


@dataclass
class SbmexParameters(MinutiaExtractionFromEnhancedImageParameters):
    """
    Parameters for the SBMEX minutia extraction algorithm.    
    """
    binarization_threshold: int = 0
    minimum_background_distance: int = 10
    minimum_tracking_distance: int = 5
    maximum_tracking_distance: int = 20
    detection_method: str = "Merge"
    merge_max_distance: int = 6
    merge_max_direction_difference: float = np.pi/6
    unpaired_minutiae_penalty: float = 0.5
    valley_minutiae_correction: float = 3
    minutiae_cluster_sigma: float = 11
    minutiae_decay_base: float = 0.5
    minutia_quality_threshold: float = 0.3


class Sbmex(MinutiaExtractionFromEnhancedImageAlgorithm):
    """
    Skeleton-based Minutiae Extraction (SBMEX) algorithm.
    """
    def __init__(self, parameters: SbmexParameters | None = None):
        if parameters is None:
            parameters = SbmexParameters()
        super().__init__(parameters)
        self.parameters = parameters

    #region Private static attributes

    # A filter that converts any 8-neighborhood configuration into a single byte value [0,255]
    _cn_filter = np.array([
            [  1,   2,   4],
            [128,   0,   8],
            [ 64,  32,  16]
        ], dtype=np.uint8)        
    
    # Lookup table mapping each encoded byte to its corresponding crossing number
    _all_8_neighborhoods = [np.array([int(b) for b in f'{x:08b}'])[::-1] for x in range(256)]
    _cn_lut = np.array([_compute_crossing_number(n) for n in _all_8_neighborhoods], dtype=np.uint8)

    _r2 = 2**0.5 # sqrt(2)

    # The eight possible (x, y) offsets with each corresponding Euclidean distance
    _xy_steps = [(-1,-1,_r2),( 0,-1,1),( 1,-1,_r2),( 1, 0,1),( 1, 1,_r2),( 0, 1,1),(-1, 1,_r2),(-1, 0,1)]

    # LUT: for each 8-neighborhood and each previous direction [0,8], 
    #      where 8 means "none", provides the list of possible directions
    _nd_lut = [[_compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in _all_8_neighborhoods]

    #endregion
    
    
    def _encode_neighborhood(self, skeleton):
        # Convert skeleton intensities from 0/255 to 0/1 binary states
        skeleton01 = (skeleton != 0).astype(np.uint8)
        # Encode the 8-neighborhood of each pixel into its respective byte representation
        return cv.filter2D(skeleton01, -1, self._cn_filter, borderType=cv.BORDER_CONSTANT)


    def _follow_ridge_and_compute_angle(self, x, y, enc_neighborhood, cn, d=8, start_x=None, start_y=None)-> tuple[float|None, float]:
        """
        Follows the skeleton until another minutia is found or a distance of maximum_tracking_distance pixels has been traveled. 
        If a minimum length of minimum_tracking_distance pixels has been reached, it returns the corresponding angle and length, otherwise it returns (None, 0).
        """
        # If starting coordinates are not provided, the pivot defaults to the current position (e.g., for Terminations)
        if start_x is None: start_x = x
        if start_y is None: start_y = y
        px, py = x, y
        length = 0.0
        while length < self.parameters.maximum_tracking_distance:
            next_directions = self._nd_lut[enc_neighborhood[py,px]][d]
            if len(next_directions) == 0:
                break
            # Check all possible next directions for another minutia
            if (any(cn[py + self._xy_steps[nd][1], px + self._xy_steps[nd][0]] != 2 for nd in next_directions)):
                break # Another minutia found: stop tracking
            # Only the first direction has to be followed
            d = next_directions[0]
            ox, oy, l = self._xy_steps[d]
            px += ox ; py += oy ; length += l
        # Compute the angle relative to the true core pivot (start_x, start_y)
        return (math.atan2(-py + start_y, px - start_x), length) if length >= self.parameters.minimum_tracking_distance else (None, 0)


    def _compute_minutiae_directions(self, minutiae: list[Minutia], enc_neighborhood, cn) -> list[Minutia]:
        valid_minutiae = []
        for x, y, _, t, _ in minutiae:
            d, length = None, 0
            if t == "E": # Ridge ending: simply follow and compute the direction
                d, length = self._follow_ridge_and_compute_angle(x, y, enc_neighborhood, cn)
            else: # Bifurcation: follow each of the three branches
                dirs = self._nd_lut[enc_neighborhood[y,x]][8] # 8 means: no previous direction
                if len(dirs)==3: # Process only if there are exactly three branches
                    angles_and_lengths = [self._follow_ridge_and_compute_angle(x+self._xy_steps[d][0], y+self._xy_steps[d][1], enc_neighborhood, cn, d, start_x=x, start_y=y) for d in dirs]
                    angles, lengths = zip(*angles_and_lengths)
                    if all(a is not None for a, _ in angles_and_lengths):
                        # Identify the two structurally closest branches out of the three
                        a1, a2 = min(((angles[i], angles[(i+1)%3]) for i in range(3)), key=lambda t: _angle_difference(t[0], t[1]))
                        # The final direction is the bisector of the two closest branches
                        d = _angle_mean(a1, a2)
                        length = sum(lengths) / 3
            if d is not None:
                q = min(1, length / self.parameters.maximum_tracking_distance) # Initial quality based on the tracking length
                valid_minutiae.append( Minutia(x, y, d, t, q) )
        return valid_minutiae


    def _process_binarized_image(self, binarized_image: Image, mask_distance, intermediate_results, intermediate_results_name) -> list[Minutia]:
        # 1. Extract the 1-pixel wide skeleton medial axis
        skeleton = cv.ximgproc.thinning(binarized_image, thinningType=cv.ximgproc.THINNING_GUOHALL)
        if intermediate_results is not None: 
            intermediate_results.append((skeleton, f'Skeleton {intermediate_results_name}'))

        # 2. Perform fast neighborhood state encoding and compute crossing numbers
        encoded_neighborhoods = self._encode_neighborhood(skeleton)
        cn = cv.LUT(encoded_neighborhoods, self._cn_lut)        
        cn[skeleton == 0] = 0 # Suppress background context: strictly preserve crossing numbers that lie on the skeleton path

        # 3. Extract coordinates for valid minutiae (cn == 1 -> Ridge ending 'E', cn == 3 -> Bifurcation 'B')
        y_coords, x_coords = np.where((cn == 1) | (cn == 3))
        minutiae = [Minutia(int(x), int(y), 0, 'E' if cn[y, x] == 1 else 'B', 1) for y, x in zip(y_coords, x_coords)]
        if intermediate_results is not None: 
            intermediate_results.append((minutiae, f'Initial minutiae {intermediate_results_name}'))

        # 4. Clean false minutiae near the segmentation perimeter
        bd_filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]] > self.parameters.minimum_background_distance, minutiae))
        if intermediate_results is not None:
            intermediate_results.append((bd_filtered_minutiae, f'Minutiae after filtering by distance from mask border {intermediate_results_name}'))

        # 5. Compute directions and initial quality scores
        minutiae_with_directions = self._compute_minutiae_directions(bd_filtered_minutiae, encoded_neighborhoods, cn)
        if intermediate_results is not None:
            intermediate_results.append((minutiae_with_directions, f'Minutiae after direction computation {intermediate_results_name}'))

        return minutiae_with_directions
        

    def _shift_minutiae(self, minutiae: list[Minutia], pixels: float) -> list[Minutia]:
        """
        Shifts coordinates. Bifurcations ("B") move opposite to the angle.
        Accounts for inverted y-axis.
        Note that minutiae types in valleys should be already inverted before calling this function.
        """
        return [m._replace(x=round(m.x + p * math.cos(m.direction)), y=round(m.y - p * math.sin(m.direction)))
                for m in minutiae if (p := -pixels if m.type == "B" else pixels) or True]
        

    def _decrease_quality_of_minutiae_clusters(self, minutiae: list[Minutia], sigma: float, decay_base: float) -> list[Minutia]:
        if len(minutiae) < 2:
            return minutiae
        coords = np.array([[m.x, m.y] for m in minutiae], dtype=np.float32)
        double_sigma_sq = 2.0 * (sigma ** 2)
        proximity = np.exp(-squareform(pdist(coords, metric='sqeuclidean')) / double_sigma_sq)
        np.fill_diagonal(proximity, 0.0)
        sum_proximity = proximity.sum(axis=1)
        penalties = np.power(decay_base, sum_proximity)
        return [m._replace(quality=m.quality * penalties[i]) if sum_proximity[i] > 1e-5 else m for i, m in enumerate(minutiae)]


    def run(self, image: Image, mask: Image, dpi: int = 500, intermediate_results : list | None = None) -> list[Minutia]:
        params = self.parameters
        _, binarized_image = cv.threshold(image, params.binarization_threshold, 255, cv.THRESH_BINARY)
        mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]

        if params.detection_method in ["Ridges", "Merge", "Merge1"]:
            binarized_image_ridges = cv.bitwise_and(binarized_image, mask)
            if intermediate_results is not None: 
                intermediate_results.append((binarized_image_ridges, 'Binarized image (ridges)'))
            ridge_minutiae = self._process_binarized_image(binarized_image_ridges, mask_distance, intermediate_results, "(ridges)")
            if intermediate_results is not None: 
                intermediate_results.append((ridge_minutiae, 'Minutiae from ridge skeleton'))
        else:
            ridge_minutiae = []

        if params.detection_method in ["Valleys", "Merge", "Merge1"]:
            binarized_image_valleys = cv.bitwise_and(cv.bitwise_not(binarized_image), mask)
            if intermediate_results is not None: 
                intermediate_results.append((binarized_image_valleys, 'Binarized image (valleys)'))
            valley_minutiae = self._process_binarized_image(binarized_image_valleys, mask_distance, intermediate_results, "(valleys)")
            valley_minutiae = [m._replace(type='E' if m.type == "B" else 'B') for m in valley_minutiae]
            valley_minutiae = self._shift_minutiae(valley_minutiae, params.valley_minutiae_correction)
            if intermediate_results is not None: 
                intermediate_results.append((valley_minutiae, 'Minutiae from valley skeleton'))
        else:
            valley_minutiae = []

        res = None
        match(params.detection_method):
            case "Ridges":
                res = ridge_minutiae
            case "Valleys":
                res = valley_minutiae
            case "Merge":
                _, _, pairs = compare_minutiae(ridge_minutiae, valley_minutiae, False, params.merge_max_distance, params.merge_max_direction_difference)
                paired_ridge_minutiae = set(i for i, _ in pairs)
                paired_valley_minutiae = set(j for _, j in pairs)
                res = [m._replace(quality=m.quality*params.unpaired_minutiae_penalty) if i not in paired_ridge_minutiae else m for i,m in enumerate(ridge_minutiae)] + \
                      [m._replace(quality=m.quality*params.unpaired_minutiae_penalty) for i,m in enumerate(valley_minutiae) if i not in paired_valley_minutiae]
            case _:
                raise ValueError("Invalid detection method")

        res = self._decrease_quality_of_minutiae_clusters(res, params.minutiae_cluster_sigma, params.minutiae_decay_base)

        # Final filtering based on quality threshold
        return [m for m in res if m.quality >= params.minutia_quality_threshold]

