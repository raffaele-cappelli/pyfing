import os
import math
import keras
import cv2 as cv
from abc import abstractmethod, ABC
from ._internal_utils import _predict_and_get_all_outputs, _resize_and_crop_intermediate_output
from .definitions import *


class FrequencyEstimationParameters(Parameters):
    """
    Base class for the parameters of a frequency estimation method.
    """
    pass


class FrequencyEstimationAlgorithm(ABC):
    """
    Base class for frequency estimation methods.
    """
    def __init__(self, parameters: FrequencyEstimationParameters):
        self.parameters = parameters
    
    @abstractmethod
    def run(self, image: Image, mask: Image, orientation_field: np.ndarray, dpi: int = 500, intermediate_results = None) -> np.ndarray:
        raise NotImplementedError
    
    def run_on_db(self, images: list[Image], masks: list[Image], orientation_fields: list[np.ndarray], dpi_of_images: list[int]) -> list[np.ndarray]:
        return [self.run(img, mask, orientation_field, dpi) for img, mask, orientation_field, dpi in zip(images, masks, orientation_fields, dpi_of_images)]



def compute_ridge_period_MAPE(rp, gt, gt_mask):
    """Returns the Mean Absolute Percentage Error (MAPE) of ridge periods with respect to gt, 
    computed only on gt_mask pixels"""
    diff = np.abs(rp - gt).astype(np.float32)
    return (diff[gt_mask != 0]*100/gt[gt_mask!=0]).mean()



class SkffeParameters(FrequencyEstimationParameters):
    """
    Parameters of SKFFE (SKeleton-based Fingerprint Frequency Estimation) method
    """

    def __init__(self, period_min = 5, period_max = 18, median_blur_size = 5, final_blur_size = 5):
        self.period_min = period_min
        self.period_max = period_max
        self.median_blur_size = median_blur_size
        self.final_blur_size = final_blur_size        
        

class Skffe(FrequencyEstimationAlgorithm):
    """
    Implementation of SKFFE (SKeleton-based Fingerprint Frequency Estimation) method
    This method should be used only on binary skeleton images, not on raw fingerprint images.
    """

    def __init__(self, parameters : SkffeParameters = None):
        if parameters is None:
            parameters = SkffeParameters()
        super().__init__(parameters)
        self.parameters = parameters
        self._offsets = self._build_offsets()

    def _build_offsets(self):
        max_len = 25
        discretized_orientations = 180
        offsets = np.empty((discretized_orientations, max_len, 2, 2), dtype = np.int32)
        for or_index in range(discretized_orientations):
            orientation = or_index * math.pi / discretized_orientations
            orientation += math.pi / 2 # orthogonal orientation
            cos = math.cos(orientation)
            sin = -math.sin(orientation) # orientation was counter-clockwise
            for i in range(max_len):
                offset_x = int(round(i*cos))
                offset_y = int(round(i*sin))
                offsets[or_index, i, 0, 0] = offset_x
                offsets[or_index, i, 0, 1] = offset_y
                offsets[or_index, i, 1, 0] = -offset_x
                offsets[or_index, i, 1, 1] = -offset_y
        return offsets


    def run(self, image: Image, mask: Image, orientation_field: np.ndarray, intermediate_results = None) -> np.ndarray:
        parameters = self.parameters
        h, w = image.shape
        res = np.zeros((h, w), np.float32)        
        for y, x in np.argwhere(image):
            orientation = int(round(orientation_field[y, x] * 180 / np.pi) % 180)
            offsets = self._offsets[orientation]
            rp1, px1, py1 = self._find_next_ridge(image, h, w, y, x, offsets, 0)
            rp2, px2, py2 = self._find_next_ridge(image, h, w, y, x, offsets, 1)
            if intermediate_results is not None: 
                rx, ry = max(0,x-25), max(0,y-25)
                roi = cv.cvtColor(image[ry:y+25, rx:x+25], cv.COLOR_GRAY2BGR)
                cv.drawMarker(roi, (x-rx, y-ry), (255,0,0), cv.MARKER_CROSS)
                if rp1 != -1:
                    cv.drawMarker(roi, (px1-rx, py1-ry), (0,255,0), cv.MARKER_CROSS)
                if rp2 != -1:
                    cv.drawMarker(roi, (px2-rx, py2-ry), (0,255,0), cv.MARKER_CROSS)
                intermediate_results.append((roi, f"{rp1} | {rp2}"))
            if rp1 != -1 and rp2 != -1:
                rp = (rp1+rp2)/2
            else:
                rp = rp1 if rp1 != -1 else rp2
            if parameters.period_min <= rp <= parameters.period_max:
                res[y, x] = rp        
        if intermediate_results is not None:
            intermediate_results.append((res, 'Initial result'))
        inpaint_mask = (res == 0).astype(np.uint8)
        res = cv.inpaint(res, inpaint_mask, 31, cv.INPAINT_NS)
        if intermediate_results is not None:
            intermediate_results.append((res, 'After inpainting'))        
        if parameters.median_blur_size>0:
            res = cv.medianBlur(res, parameters.median_blur_size)
        if parameters.final_blur_size>0:
            res = cv.GaussianBlur(res, (parameters.final_blur_size, parameters.final_blur_size), 0)
        res[mask == 0] = 0        
        return res 


    def _find_next_ridge(self, image, h, w, y, x, offsets, direction):
        offsets_count = self._offsets.shape[1]
        rp = -1
        last_px, last_py = None, None
        for i in range(3, offsets_count):
            px, py = x + offsets[i, direction, 0], y + offsets[i, direction, 1]
            if 0<=px<w and 0<=py<h:
                if image[py, px] != 0 or (last_px is not None and (abs(last_px-px) == 1 and abs(last_py-py)==1) and image[last_py, px] != 0):
                    rp = i
                    break
            last_px, last_py = px, py
        return rp,px,py
    

class XsffeParameters(FrequencyEstimationParameters):
    """
    Parameters of XSFFE (X-Signature Fingerprint Frequency Estimation) method
    """

    def __init__(self, window_size = (23, 43), step = 8, border = 7, min_background_distance = 11, period_min = 5, period_max = 20,
                 min_valid_distances = 4, diffusion_size = 21, median_size = 5, blur_size = 3, final_blur_size = 33):
        self.window_size = window_size
        self.step = step
        self.border = border
        self.min_background_distance = min_background_distance
        self.min_valid_distances = min_valid_distances
        self.period_min = period_min
        self.period_max = period_max
        self.median_size = median_size
        self.blur_size = blur_size
        self.final_blur_size = final_blur_size
        self.diffusion_size = diffusion_size


class Xsffe(FrequencyEstimationAlgorithm):
    """
    Implementation of XSFFE (X-Signature Fingerprint Frequency Estimation) method.
    """
    
    def __init__(self, parameters : XsffeParameters = None):
        if parameters is None:
            parameters = XsffeParameters()
        super().__init__(parameters)
        self.parameters = parameters

    def run(self, image: Image, mask: Image, orientation_field: np.ndarray, dpi: int = 500, intermediate_results = None) -> np.ndarray:
        if dpi != 500:
            raise ValueError("Only 500 dpi images are currently supported")
        parameters = self.parameters                   
        wnd_hw, wnd_hh = parameters.window_size[0]//2, parameters.window_size[1]//2    
        h, w = image.shape
        border = parameters.border
        nw, nh = ((w-border*2) // parameters.step) + 1, ((h-border*2) // parameters.step) + 1
        if parameters.median_size > 0:
            image = cv.medianBlur(image, parameters.median_size)
            if intermediate_results is not None: 
                intermediate_results.append((image, 'After Median blur'))
        if parameters.blur_size > 0:
            image = cv.GaussianBlur(image, (parameters.blur_size, parameters.blur_size), 0)
            if intermediate_results is not None: 
                intermediate_results.append((image, 'After Gaussian blur'))    
        mask_distance = cv.distanceTransform(cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]
        if intermediate_results is not None: 
            img_points = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        res = np.zeros((nh, nw), np.float32)
        angles = 180-np.degrees(orientation_field)
        for iy, y in enumerate(range(border, h-border, parameters.step)):
            for ix, x in enumerate(range(border, w-border, parameters.step)):
                if mask_distance[y, x] >= parameters.min_background_distance:
                    M = cv.getRotationMatrix2D((x,y), angles[y,x], 1.0)
                    M[0, 2] += wnd_hw - x
                    M[1, 2] += wnd_hh - y                  
                    region = cv.warpAffine(image, M, parameters.window_size, flags = cv.INTER_LINEAR, borderMode = cv.BORDER_REFLECT)
                    xs = np.sum(region, 1) # the x-signature of the region
                    # Find the indices of the x-signature local maxima / minima
                    xs1 = xs[1:-1] 
                    local_maxima = 1 + np.nonzero((xs1 > xs[:-2]) & (xs1 >= xs[2:]))[0] # Valleys
                    local_minima = 1 + np.nonzero((xs1 < xs[:-2]) & (xs1 <= xs[2:]))[0] # Ridges
                    if local_maxima.size + local_minima.size >= 2 + parameters.min_valid_distances:
                        # Calculate the median of all the valid distances between consecutive peaks / valleys
                        valid_dists = np.concatenate((local_maxima[1:] - local_maxima[:-1], local_minima[1:] - local_minima[:-1]))
                        valid_dists = valid_dists[(valid_dists >= parameters.period_min) & (valid_dists <= parameters.period_max)]
                        if valid_dists.size >= parameters.min_valid_distances:
                            mv = np.median(valid_dists)
                            close_dists = valid_dists[np.abs(valid_dists-mv)<=2]
                            if close_dists.size > 0:
                                res[iy, ix] = close_dists.mean()            
                                if intermediate_results is not None:
                                    cv.drawMarker(img_points, (x, y), (255,0,0))
                                    intermediate_results.append((self._plot_x_signature(region, xs, local_maxima), f'[{mask_distance[y, x]}] ({ix},{iy}): ({x},{y}) [{angles[y,x]:.1f}] = {res[iy, ix]:.2f}'))
        if intermediate_results is not None:
            intermediate_results.append((img_points, 'Sampling positions'))
            intermediate_results.append((res, 'Sampling result'))

        # Finally create the result with the same size of fingerprint by resising res        
        inpaint_mask = (res == 0).astype(np.uint8)
        res = cv.inpaint(res, inpaint_mask, parameters.diffusion_size / parameters.step, cv.INPAINT_NS)
        if intermediate_results is not None:
            intermediate_results.append((res, 'After inpainting'))

        r_w, r_h = parameters.step*nw, parameters.step*nh
        res = cv.resize(res, (r_w, r_h), interpolation = cv.INTER_LINEAR)
        b1 = border - 1 - parameters.step // 2
        
        res = cv.copyMakeBorder(res, b1, h - r_h - b1, b1, w - r_w - b1, cv.BORDER_CONSTANT)
        inpaint_mask = (res == 0).astype(np.uint8)
        res = cv.inpaint(res, inpaint_mask, parameters.diffusion_size, cv.INPAINT_NS)
        
        if intermediate_results is not None:
            intermediate_results.append((res, 'After resize'))
        if parameters.final_blur_size>0:
            res = cv.GaussianBlur(res, (parameters.final_blur_size, parameters.final_blur_size), 0)

        res[mask == 0] = 0        
        return res        

    def _plot_x_signature(self, region, xs, maxima):
        size = 100
        rw = region.shape[1]
        rh = region.shape[0]
        img = np.zeros((xs.size, rw+size+1, 3), np.uint8)
        points = np.array([(rw + int(x*size/(255*rh)), y) for y, x in enumerate(xs)])
        cv.polylines(img, [points], False, (255,255,255), 1, cv.LINE_AA)
        
        for y in maxima:
            img[y, size, 0] = 255
        img[:,size+1:,:] = region[...,np.newaxis]
        return img
    

    
class SnffeParameters(FrequencyEstimationParameters):
    """
    Parameters of SNFFE (Simple Network for Fingerprint Frequency Estimation) method.
    """
    def __init__(self, dnn_input_dpi = 500, dnn_input_size_multiple = 32):
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple


class Snffe(FrequencyEstimationAlgorithm):
    """
    Implementation of SNFFE (Simple Network for Fingerprint Frequency Estimation) method.
    If both model_weights and model are None, the default model installed with the package is loaded.
    """
    
    def __init__(self, parameters : SnffeParameters = None, model_weights = None, model = None):
        if parameters is None:
            parameters = SnffeParameters()
        super().__init__(parameters)
        self.parameters = parameters
        if model_weights is None and model is None:
            model_weights = os.path.dirname(__file__) + "/models/SNFFE.weights.h5"
        if model_weights is not None:
            self.model = self._build_model()
            self.model.load_weights(model_weights)
        elif model is not None:
            self.model = model
    

    def _build_model(self):
        layers = keras.layers
        input = layers.Input((None, None, 3), name="input")
        x = input        

        # Stem: replace orientation channel with double-angle-representation channels
        rad = layers.Lambda(lambda t: t[...,2:] / 255 * np.pi * 2, name="rad")(x)
        sen2 = layers.Lambda(lambda t: keras.ops.sin(t), name="sin2")(rad)
        cos2 = layers.Lambda(lambda t: keras.ops.cos(t), name="cos2")(rad)
        x = layers.Concatenate(name="concat_input")([x[...,0:2], sen2, cos2])
        
        # Encoder
        level_count = 5
        intermediate_outputs = []        
        for i in range(level_count):
            x = layers.Conv2D(16*2**i, 5, padding="same", activation = "relu", name=f"enc_{i}_conv_5")(x)
            x = layers.BatchNormalization(name=f"enc_{i}_bn")(x)
            intermediate_outputs.append(x)
            x = layers.MaxPooling2D(2, padding="same", name=f"enc_{i}_dn_maxpool_2")(x)

        # Decoder
        for i, intermediate_output in enumerate(reversed(intermediate_outputs)):
            x = layers.Conv2D(16*2**(level_count-1-i), 5, padding="same", activation = "relu", name=f"dec_{i}_conv_5")(x)
            x = layers.BatchNormalization(name=f"dec_{i}_bn")(x)
            x = layers.UpSampling2D(2, name=f"dec_{i}_up_2")(x)
            x = layers.Concatenate(name=f"dec_{i}_skip_connection")([x, intermediate_output])    

        # Head
        x = layers.Conv2D(16, 5, padding="same", activation="relu", name="head_0_conv_5")(x)
        x = layers.BatchNormalization(name="head_0_bn")(x)
        x = layers.Conv2D(1, 3, activation="linear", padding="same", name="head_linear_conv_3")(x)

        return keras.Model(input, x[...,0])


    def run(self, image: Image, mask: Image, orientation_field: np.ndarray, dpi: int = 500, intermediate_results = None) -> np.ndarray:
        parameters = self.parameters
        original_image_h, original_image_w = image.shape

        if dpi != parameters.dnn_input_dpi:
            f = parameters.dnn_input_dpi / dpi
            image = cv.resize(image, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)
            mask = cv.resize(mask, None, fx = f, fy = f, interpolation = cv.INTER_NEAREST)
            cos2, sin2 = np.cos(orientation_field*2), np.sin(orientation_field*2)
            cos2 = cv.resize(cos2, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)
            sin2 = cv.resize(sin2, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)       
            orientation_field = np.arctan2(sin2, cos2)/2

        h, w = image.shape    
        size_mult = parameters.dnn_input_size_multiple
        input_w, input_h = (w+size_mult-1)//size_mult*size_mult, (h+size_mult-1)//size_mult*size_mult
        border_left, border_top = (input_w-w)//2, (input_h-h)//2
        border_right, border_bottom = input_w-w-border_left, input_h-h-border_top
        ir = cv.copyMakeBorder(image, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT, value = image[0,0].tolist())
        mr = cv.copyMakeBorder(mask//255, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        orr = cv.copyMakeBorder(orientation_field, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        orr = np.round((orr % np.pi) * 255 / np.pi).clip(0,255).astype(np.uint8)
        if intermediate_results is not None:
            if dpi != parameters.dnn_input_dpi: 
                raise NotImplementedError("Intermediate results are not available for different input resolution")
            res = _predict_and_get_all_outputs(self.model, np.dstack((ir, mr, orr))[np.newaxis,...])
            intermediate_results += [(_resize_and_crop_intermediate_output(w, h, border_left, border_top, border_right, border_bottom, r), l) for r, l in res]
            rp = res[-1][0][0]
        else:
            # From keras documentation: "For small numbers of inputs that fit in one batch, directly use __call__() for faster execution"
            rp = self.model(np.dstack((ir, mr, orr))[np.newaxis,...], training = False).numpy()[0] 
        rp = rp[border_top:border_top+h, border_left:border_left+w]

        if dpi != parameters.dnn_input_dpi:
            rp = cv.resize(rp, (original_image_w, original_image_h), interpolation = cv.INTER_CUBIC)
            rp *= (dpi / parameters.dnn_input_dpi)

        rp /= 10
        return rp
    
