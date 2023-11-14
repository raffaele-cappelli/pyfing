from abc import abstractmethod, ABC
import math
import numpy as np
import cv2 as cv
import tensorflow as tf
from .definitions import Image, Parameters


class SegmentationParameters(Parameters):
    """
    Base class for the parameters of a segmentation method.
    """
    pass


class SegmentationAlgorithm(ABC):
    """
    Base class for segmentation methods.
    """
    def __init__(self, parameters: SegmentationParameters):
        self.parameters = parameters
    
    @abstractmethod
    def run(self, image: Image, intermediate_results = None) -> Image:
        raise NotImplementedError
    
    def run_on_db(self, images: [Image]) -> [Image]:
        return [self.run(img) for img in images]


def compute_segmentation_error(mask, gt_mask):
    """Returns the segmentation error (percentage of wrong pixels) of mask with respect to ground truth mask gt_mask"""
    return np.count_nonzero(gt_mask - mask) / mask.size


def compute_dice_coefficient(mask, gt_mask):
    """Returns the Dice Coefficient of mask with respect to ground truth mask gt_mask"""
    return 2 * np.count_nonzero(gt_mask & mask) / (np.count_nonzero(gt_mask) + np.count_nonzero(mask))

def compute_jaccard_coefficient(mask, gt_mask):
    """Returns the Jaccard similarity coefficient of mask with respect to ground truth mask gt_mask"""
    return np.count_nonzero(gt_mask & mask) / np.count_nonzero(gt_mask | mask)



class GmfsParameters(SegmentationParameters):
    """
    Parameters of the GMFS segmentation method.
    """

    def __init__(self, sigma = 13/3, percentile = 95, threshold = 0.2, closing_count = 6, opening_count = 12, image_dpi = 500):
        self.sigma = sigma
        self.percentile = percentile
        self.threshold = threshold
        self.closing_count = closing_count
        self.opening_count = opening_count
        self.image_dpi = image_dpi


class Gmfs(SegmentationAlgorithm):
    """
    The GMFS segmentation method.
    """

    def __init__(self, parameters : GmfsParameters = None):
        if parameters is None:
            parameters = GmfsParameters()
        super().__init__(parameters)
        self.parameters = parameters

    def run(self, image: Image, intermediate_results = None) -> Image:
        parameters = self.parameters

        # Resizes the image if its resolution is not 500 dpi
        image_h, image_w = image.shape
        if parameters.image_dpi != 500:
            f = 500 / parameters.image_dpi
            image = cv.resize(image, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)

        # Calculates the gradient magnitude
        gx, gy = cv.spatialGradient(image)
        sm = cv.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        if intermediate_results is not None:
            intermediate_results.append((sm, 'Gradient magnitude'))

        # Averages the gradient magnitude with a Gaussian filter
        gs =  math.ceil(3 * parameters.sigma) * 2 + 1
        r = cv.GaussianBlur(sm, (gs, gs), parameters.sigma)
        if intermediate_results is not None:
            intermediate_results.append((r, 'Average gradient magnitude'))

        # Compute the actual threshold
        norm_t = np.percentile(sm, parameters.percentile) * parameters.threshold

        # Selects pixels with average gradient magnitude above the threshold
        mask = cv.threshold(r, norm_t, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
        if intermediate_results is not None:
            intermediate_results.append((np.copy(mask), 'Thresholding'))

        if parameters.closing_count > 0:
            # Applies closing to fill small holes and concavities
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self._se3x3, iterations = parameters.closing_count)
            if intermediate_results is not None:
                intermediate_results.append((np.copy(mask), 'After closing'))

        # Remove all but the largest component
        self._remove_other_cc(mask)
        if intermediate_results is not None:
            intermediate_results.append((np.copy(mask), 'Largest component'))

        # Use connected components labeling to fill holes (except those connected to the image border)
        _, cc, stats, _ = cv.connectedComponentsWithStats(cv.bitwise_not(mask)) # in the background
        h, w = image.shape
        holes = np.where((stats[:,cv.CC_STAT_LEFT] > 0) &
                         (stats[:,cv.CC_STAT_LEFT] + stats[:,cv.CC_STAT_WIDTH] < w-1) &
                         (stats[:,cv.CC_STAT_TOP] > 0) &
                         (stats[:,cv.CC_STAT_TOP] + stats[:,cv.CC_STAT_HEIGHT] < h-1))
        mask[np.isin(cc, holes)] = 255
        if intermediate_results is not None:
            intermediate_results.append((np.copy(mask), 'After fill holes'))

        if parameters.opening_count > 0:
            # Applies opening to remove small blobs and protrusions
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self._se3x3, iterations = parameters.opening_count, borderValue = 0)
            if intermediate_results is not None:
                intermediate_results.append((np.copy(mask), 'After opening'))

        # The previous step may have created more cc: keep only the largest
        self._remove_other_cc(mask)

        if parameters.image_dpi != 500:
            mask = cv.resize(mask, (image_w, image_h), interpolation = cv.INTER_NEAREST)

        return mask


    def _remove_other_cc(self, mask):
        num, cc, stats, _ = cv.connectedComponentsWithStats(mask)
        if num > 1:
            index = np.argmax(stats[1:,cv.CC_STAT_AREA]) + 1
            mask[cc!=index] = 0


    _se3x3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))



class SufsParameters(SegmentationParameters):  
    """
    Parameters of the SUFS segmentation method.
    """
    
    def __init__(self, model_name = "SUFS.h5", dnn_input_dpi = 500, dnn_input_size = (512, 512), image_dpi = 500, threshold = 0.5):
        self.model_name = model_name        
        self.dnn_input_size = dnn_input_size
        self.dnn_input_dpi = dnn_input_dpi
        self.image_dpi = image_dpi
        self.threshold = threshold


class Sufs(SegmentationAlgorithm):
    """
    The SUFS segmentation method.
    """

    def __init__(self, parameters : SufsParameters = None, models_folder = "./models/"):
        if parameters is None:
            parameters = SufsParameters()
        super().__init__(parameters)
        self.parameters = parameters
        self.models_folder = models_folder
        if self.parameters.model_name != "":
            self.load_model()

    def load_model(self):
        """
        Loads the keras model parameters.model_name.
        """
        self.model = tf.keras.models.load_model(self.models_folder + self.parameters.model_name)

    def run(self, image: Image, intermediate_results = None) -> Image:
        h, w = image.shape
        ir = self._adjust_input(image)
        if intermediate_results is not None: intermediate_results.append((np.copy(ir), 'Adjusted input'))
        val_preds = self.model.predict(ir[np.newaxis,...,np.newaxis],verbose = 0)
        if intermediate_results is not None: intermediate_results.append((np.copy(val_preds[0].squeeze()), 'Net output'))
        mask = np.where(val_preds[0].squeeze()<self.parameters.threshold, 0, 255).astype(np.uint8) # [0..1] ==> 0,255
        return self._adjust_output(mask, (w, h))

    def run_on_db(self, images: [Image]) -> [Image]:
        ar = np.array([self._adjust_input(image)[...,np.newaxis] for image in images])
        masks = np.where(self.model.predict(ar, verbose = 0)<self.parameters.threshold,0,255).astype(np.uint8) # [0..1] ==> 0,255
        return [self._adjust_output(mask.squeeze(), image.shape[::-1]) for mask,image in zip(masks, images)]

    def _adjust_input(self, image):
        if self.parameters.dnn_input_dpi != self.parameters.image_dpi:
            # Resize to make its resolution parameters.dnn_input_dpi
            f = self.parameters.dnn_input_dpi / self.parameters.image_dpi
            image = cv.resize(image, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)
        return _adjust_size(image, self.parameters.dnn_input_size, image[0,0].tolist())

    def _adjust_output(self, mask, size):
        if self.parameters.dnn_input_dpi != self.parameters.image_dpi:
            # Resize to target resolution
            f = self.parameters.image_dpi / self.parameters.dnn_input_dpi
            mask = cv.resize(mask, None, fx = f, fy = f, interpolation = cv.INTER_NEAREST)
        # Crop or add borders
        return _adjust_size(mask, size, 0)


def _adjust_size(image, target_size, border_value):
    # For each side computes crop size (if negative) or border to be added (if positive)
    h, w = image.shape
    target_w, target_h = target_size
    left = (target_w - w) // 2
    right = target_w - w - left
    top = (target_h - h) // 2
    bottom = target_h - h - top
    if left < 0 or right < 0: # Horizontal crop
        image = image[:, -left:(right if right < 0 else w)]
    if top < 0 or bottom < 0: # Vertical crop
        image = image[-top:(bottom if bottom < 0 else h)]    
    if left > 0 or right > 0 or top > 0 or bottom > 0: # Add borders
        image = cv.copyMakeBorder(image, max(0,top), max(0,bottom), max(0,left), max(0,right), cv.BORDER_CONSTANT, value = border_value)
    return image


