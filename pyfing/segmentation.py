from abc import abstractmethod, ABC
import os
import keras
import math
import numpy as np
import cv2 as cv
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
        m = cv.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        if intermediate_results is not None:
            intermediate_results.append((m, 'Gradient magnitude'))

        # Averages the gradient magnitude with a Gaussian filter
        gs =  math.ceil(3 * parameters.sigma) * 2 + 1
        m_a = cv.GaussianBlur(m, (gs, gs), parameters.sigma)
        if intermediate_results is not None:
            intermediate_results.append((m_a, 'Average gradient magnitude'))

        # Compute the actual threshold
        norm_t = np.percentile(m, parameters.percentile) * parameters.threshold

        # Selects pixels with average gradient magnitude above the threshold
        mask = cv.threshold(m_a, norm_t, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
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
    
    def __init__(self, dnn_input_dpi = 500, dnn_input_size_multiple = 64, image_dpi = 500, threshold = 0.5, border = 33):
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple
        self.image_dpi = image_dpi
        self.threshold = threshold
        self.border = border


class Sufs(SegmentationAlgorithm):
    """
    The SUFS segmentation method.
    If both model_weights and model are None, the default model installed with the package is loaded.
    """
    def __init__(self, parameters : SufsParameters = None, model_weights = None, model = None):
        if parameters is None:
            parameters = SufsParameters()
        super().__init__(parameters)
        self.parameters = parameters
        if model_weights is None and model is None:
            model_weights = os.path.dirname(__file__) + "/models/SUFS.weights.h5"
        if model_weights is not None:
            self.model = self._build_model()
            self.model.load_weights(model_weights)
        elif model is not None:
            self.model = model


    def _build_model(self):
        FILTERS = [16, 32, 64, 128, 256, 512]
        layers = keras.layers
        inputs = layers.Input((None, None, 1))
        x = inputs    
        level_outputs = []
        for filters in FILTERS:
            x = layers.Conv2D(filters, 3, padding="same", activation = "relu")(x)
            x = layers.BatchNormalization()(x)
            level_outputs.append(x)
            x = layers.MaxPooling2D(2, padding="same")(x)
        for filters, lo in zip(reversed(FILTERS), reversed(level_outputs)):        
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling2D(2)(x)
            x = layers.Concatenate()([x, lo])
        outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
        return keras.Model(inputs, outputs)


    def run(self, image: Image, intermediate_results = None) -> Image:
        original_image_h, original_image_w = image.shape
        size_info = self._compute_size(original_image_w, original_image_h)        
        ir = self._adjust_input(image, *size_info)
        if intermediate_results is not None: intermediate_results.append((np.copy(ir), 'Adjusted input'))
        val_preds = self.model.predict(ir[np.newaxis,...,np.newaxis],verbose = 0)
        if intermediate_results is not None: intermediate_results.append((np.copy(val_preds[0].squeeze()), 'Net output'))
        mask = np.where(val_preds[0].squeeze()<self.parameters.threshold, 0, 255).astype(np.uint8) # [0..1] ==> 0,255
        return self._adjust_output(mask, *size_info, original_image_w, original_image_h)


    def run_on_db(self, images: [Image]) -> [Image]:
        # N.B. Assumes all the images have the same size!
        original_image_h, original_image_w = images[0].shape
        size_info = self._compute_size(original_image_w, original_image_h)
        res = []
        batch_size = 256
        for i in range(0, len(images), batch_size):
            ar = np.array([self._adjust_input(image, *size_info)[...,np.newaxis] for image in images[i:i+batch_size]])
            masks = np.where(self.model.predict(ar, verbose = 0)<self.parameters.threshold,0,255).astype(np.uint8) # [0..1] ==> 0,255
            res += [self._adjust_output(mask.squeeze(), *size_info, original_image_w, original_image_h) for mask in masks]
        return res


    def _compute_size(self, original_w, original_h):
        w, h = original_w, original_h
        if self.parameters.dnn_input_dpi != self.parameters.image_dpi:
            # Resize to make its resolution parameters.dnn_input_dpi
            f = self.parameters.dnn_input_dpi / self.parameters.image_dpi
            w, h = int(round(original_w * f)), int(round(original_h * f))
        size_mult = self.parameters.dnn_input_size_multiple
        input_w, input_h = (w+self.parameters.border+size_mult-1)//size_mult*size_mult, (h+self.parameters.border+size_mult-1)//size_mult*size_mult
        border_left, border_top = (input_w-w)//2, (input_h-h)//2
        border_right, border_bottom = input_w-w-border_left, input_h-h-border_top
        return w, h, border_left, border_top, border_right, border_bottom


    def _adjust_input(self, image, w, h, border_left, border_top, border_right, border_bottom):
        original_h, original_w = image.shape
        if w != original_w or h != original_h:
            image = cv.resize(image, (w, h), interpolation = cv.INTER_CUBIC)
        return cv.copyMakeBorder(image, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT, value = image[0,0].tolist())


    def _adjust_output(self, mask, w, h, border_left, border_top, border_right, border_bottom, original_image_w, original_image_h):
        mask = mask[border_top:border_top+h, border_left:border_left+w]
        if w != original_image_w or h != original_image_h:
            mask = cv.resize(mask, (original_image_w, original_image_h), interpolation = cv.INTER_NEAREST)        
        return mask