import os
import math
import keras
import cv2 as cv
from abc import abstractmethod, ABC
from ._internal_utils import _predict_and_get_all_outputs, _resize_and_crop_intermediate_output
from .definitions import *

class EnhancementParameters(Parameters):
    """
    Base class for the parameters of an enhancement method.
    """
    pass


class EnhancementAlgorithm(ABC):
    """
    Base class for enhancement methods.
    """
    def __init__(self, parameters: EnhancementParameters):
        self.parameters = parameters
    
    @abstractmethod
    def run(self, image: Image, mask: Image, orientation_field: np.ndarray, ridge_periods: np.ndarray, dpi: int = 500, intermediate_results = None) -> Image:
        raise NotImplementedError
    
    def run_on_db(self, images: list[Image], masks: list[Image], orientation_fields: list[np.ndarray], ridge_periods: list[np.ndarray], dpi_of_images: list[int]) -> list[Image]:
        return [self.run(img, mask, orientation_field, rp, dpi) for img, mask, orientation_field, rp, dpi in zip(images, masks, orientation_fields, ridge_periods, dpi_of_images)]

 


class GbfenParameters(EnhancementParameters):
    """
    Parameters of GBFEN (Gabor-Based Fingerprint ENhancement) method.
    """
    def __init__(self, orientations_count = 16, periods_count = 9, period_min = 5, period_max = 20):
        self.orientations_count = orientations_count
        self.periods_count = periods_count
        self.period_min = period_min
        self.period_max = period_max


class Gbfen(EnhancementAlgorithm):
    """
    Implementation of GBFEN (Gabor-Based Fingerprint ENhancement) method.
    """    
    def __init__(self, parameters : GbfenParameters = None):
        if parameters is None:
            parameters = GbfenParameters()
        super().__init__(parameters)
        self.parameters = parameters
        self._gabor_bank = [self._gabor_kernel(rp, o) 
            for rp in np.linspace(parameters.period_min, parameters.period_max, parameters.periods_count)
                for o in np.arange(0, np.pi, np.pi/parameters.orientations_count)]
        
    def _gabor_kernel(self, period, orientation):
        sigma = 5 * period / 12
        s = int(math.ceil(3 * sigma)) * 2 + 1
        f = cv.getGaborKernel((s, s), sigma, np.pi/2 - orientation, period, gamma = 1.0, psi = 0, ktype = cv.CV_32F)
        f -= f.mean()
        return f / np.sqrt((f**2).sum())
    
    def _discretize_ridge_periods(self, periods, period_min, period_max, period_count):
        return np.round((np.clip(periods, period_min, period_max) - period_min) / (period_max - period_min) * (period_count-1)).astype(np.int32)
    
    def _discretize_orientations(self, orientations, orientation_count):
        return np.round(((orientations % np.pi) / np.pi) * orientation_count).astype(np.int32) % orientation_count
    
    def _gabor_bank_indices(self, orientations, ridge_periods, orientations_count, period_min, period_max, period_count):
        return self._discretize_ridge_periods(ridge_periods, period_min, period_max, period_count) * orientations_count + \
            self._discretize_orientations(orientations, orientations_count)    

    def run(self, image: Image, mask: Image, orientation_field: np.ndarray, ridge_periods: np.ndarray, dpi: int = 500, intermediate_results = None) -> Image:
        p = self.parameters
        original_image_h, original_image_w = image.shape
        if intermediate_results is not None: intermediate_results += [(f, f'f{i}') for i, f in enumerate(self._gabor_bank)]            
        if dpi != 500:
            f = 500 / dpi
            image = cv.resize(image, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)
        gbi = self._gabor_bank_indices(orientation_field, ridge_periods, p.orientations_count, p.period_min, p.period_max, p.periods_count)            
        needed = np.zeros(len(self._gabor_bank))
        needed[np.unique(gbi)] = 1
        z = np.zeros_like(image)        
        img = cv.bitwise_not(image) # We want white ridge-lines on a black background
        # Unfortunately, contextual convolution is not available in OpenCV: we apply all needed filters to the whole image
        r = np.array([cv.filter2D(img, -1, f) if v==1 else z for f, v in zip(self._gabor_bank, needed)])        
        if intermediate_results is not None: intermediate_results += [(f, f'{j}') for j, f in enumerate(r)]
        y, x = np.indices(img.shape)
        img = r[gbi, y, x]
        if dpi != 500:
            img = cv.resize(img, (original_image_w, original_image_h), interpolation = cv.INTER_CUBIC)
        return img


    
class SnfenParameters(EnhancementParameters):
    """
    Parameters of SNFEN (Simple Network for Fingerprint ENhancement) method.
    """
    def __init__(self, dnn_input_dpi = 500, dnn_input_size_multiple = 32):
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple


class Snfen(EnhancementAlgorithm):
    """
    Implementation of SNFEN (Simple Network for Fingerprint ENhancement) method.
    If both model_weights and model are None, the default model installed with the package is loaded.
    """
    
    def __init__(self, parameters : SnfenParameters = None, model_weights = None, model = None):
        if parameters is None:
            parameters = SnfenParameters()
        super().__init__(parameters)
        self.parameters = parameters
        if model_weights is None and model is None:
            model_weights = os.path.dirname(__file__) + "/models/SNFEN.weights.h5"
        if model_weights is not None:
            self.model = Snfen._build_model()
            self.model.load_weights(model_weights)
        elif model is not None:
            self.model = model
  

    def _build_model():
        layers = keras.layers
        level_count = 5
        filter_size = 5
        base_filter_count = 16
        input = layers.Input((None, None, 4), name="input") # fingerprint, mask, orientations, ridge-periods
        x = input        
        
        # Stem: the orientation channel is replaced by the double-angle-representation channels
        fingerprints = x[...,0:1]
        masks = x[...,1:2]
        rad2 = layers.Lambda(lambda t: keras.ops.cast(t[...,2:3], "float32") / 255 * np.pi * 2, name="rad")(x)
        sen2 = layers.Lambda(lambda t: keras.ops.sin(t), name="sin2")(rad2)
        cos2 = layers.Lambda(lambda t: keras.ops.cos(t), name="cos2")(rad2)
        rp = keras.ops.cast(x[...,3:4], "float32") / 10 # ridge period was stored as rp*10
        x = layers.Concatenate(name="concat_input")([fingerprints, masks, sen2, cos2, rp]) 
        
        # Encoder    
        intermediate_outputs = []        
        for i in range(level_count):
            x = layers.Conv2D(base_filter_count*2**i, filter_size, padding="same", activation = "relu", name=f"enc_{i}_conv")(x)
            x = layers.BatchNormalization(name=f"enc_{i}_bn")(x)
            intermediate_outputs.append(x)
            x = layers.MaxPooling2D(2, padding="same", name=f"enc_{i}_dn_maxpool_2")(x)

        # Decoder
        for i, intermediate_output in enumerate(reversed(intermediate_outputs)):
            x = layers.Conv2D(base_filter_count*2**(level_count-1-i), filter_size, padding="same", activation = "relu", name=f"dec_{i}_conv")(x)
            x = layers.BatchNormalization(name=f"dec_{i}_bn")(x)
            x = layers.UpSampling2D(2, name=f"dec_{i}_up_2")(x)
            x = layers.Concatenate(name=f"dec_{i}_skip_connection")([x, intermediate_output])    

        # Head
        x = layers.Conv2D(16, filter_size, padding="same", activation="relu", name="head_conv")(x)
        x = layers.BatchNormalization(name="head_bn")(x)
        x = layers.Conv2D(1, 5, activation="sigmoid", padding="same", name="head_conv_final")(x)
        return keras.Model(input, x[...,0])


    def run(self, image: Image, mask: Image, orientation_field: np.ndarray, ridge_periods: np.ndarray, dpi: int = 500, intermediate_results = None) -> Image:        
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
            ridge_periods = cv.resize(ridge_periods, (original_image_w, original_image_h), interpolation = cv.INTER_CUBIC)
            ridge_periods *= f

        h, w = image.shape    
        size_mult = parameters.dnn_input_size_multiple
        input_w, input_h = (w+size_mult-1)//size_mult*size_mult, (h+size_mult-1)//size_mult*size_mult
        border_left, border_top = (input_w-w)//2, (input_h-h)//2
        border_right, border_bottom = input_w-w-border_left, input_h-h-border_top
        ir = cv.copyMakeBorder(image, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT, value = image[0,0].tolist())
        mr = cv.copyMakeBorder(mask//255, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        orr = cv.copyMakeBorder(orientation_field, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        orr = np.round((orr % np.pi) * 255 / np.pi).clip(0,255).astype(np.uint8)
        rpr = cv.copyMakeBorder(ridge_periods, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT)
        rpr = np.round(rpr * 10).clip(0,255).astype(np.uint8)

        if intermediate_results is not None:
            if dpi != parameters.dnn_input_dpi: 
                raise NotImplementedError("Intermediate results are not available for different input resolution")
            res = _predict_and_get_all_outputs(self.model, np.dstack((ir, mr, orr, rpr))[np.newaxis,...])
            intermediate_results += [(_resize_and_crop_intermediate_output(w, h, border_left, border_top, border_right, border_bottom, r), l) for r, l in res]
            en = res[-1][0][0]
        else:
            # From keras documentation: "For small numbers of inputs that fit in one batch, directly use __call__() for faster execution"
            en = self.model(np.dstack((ir, mr, orr, rpr))[np.newaxis,...], training = False).numpy()[0] 
        en = en[border_top:border_top+h, border_left:border_left+w]
        en = np.clip(np.round(en*255), 0, 255).astype(np.uint8)

        if dpi != parameters.dnn_input_dpi:
            en = cv.resize(en, (original_image_w, original_image_h), interpolation = cv.INTER_CUBIC)

        return en

    def run_on_db(self, images: list[Image], masks: list[Image], orientation_fields: list[np.ndarray], ridge_periods: list[np.ndarray], dpi_of_images = None) -> list[Image]:
        if dpi_of_images is not None and any(dpi != self.parameters.dnn_input_dpi for dpi in dpi_of_images):
            raise Exception(f"Only {self.parameters.dnn_input_dpi} DPI is supported")
        batch_size = 16
        res = []
        for start_index in range(0, len(images), batch_size):
            batch_images = images[start_index:start_index+batch_size]
            n = len(batch_images)
            max_h, max_w = max(img.shape[0] for img in batch_images), max(img.shape[1] for img in batch_images)
            size_mult = self.parameters.dnn_input_size_multiple
            input_w, input_h = (max_w+size_mult-1)//size_mult*size_mult, (max_h+size_mult-1)//size_mult*size_mult
            net_input = np.empty((n, input_h, input_w, 4), np.uint8)
            border_info = []
            for k in range(n):
                image = images[start_index+k]
                mask = masks[start_index+k]
                orientations = orientation_fields[start_index+k]
                ridge_p = ridge_periods[start_index+k]
                h, w = image.shape
                border_left, border_top = (input_w-w)//2, (input_h-h)//2
                border_info.append((w, h, border_left, border_top))
                net_input[k,...,0] = cv.copyMakeBorder(image, border_top, input_h-h-border_top, border_left, input_w-w-border_left, cv.BORDER_CONSTANT, value = image[0,0].tolist())
                net_input[k,...,1] = cv.copyMakeBorder(mask//255, border_top, input_h-h-border_top, border_left, input_w-w-border_left, cv.BORDER_CONSTANT)
                orr = cv.copyMakeBorder(orientations, border_top, input_h-h-border_top, border_left, input_w-w-border_left, cv.BORDER_CONSTANT)
                net_input[k,...,2] = np.round((orr % np.pi) * 255 / np.pi).clip(0,255).astype(np.uint8)
                rpr = cv.copyMakeBorder(ridge_p, border_top, input_h-h-border_top, border_left, input_w-w-border_left, cv.BORDER_CONSTANT)
                net_input[k,...,3] = np.round(rpr * 10).clip(0,255).astype(np.uint8)             
            batch_res = self.model.predict(net_input, verbose = 0)
            batch_res = np.clip(np.round(batch_res*255), 0, 255).astype(np.uint8)
            for k in range(n):
                w, h, border_left, border_top = border_info[k]
                res.append(batch_res[k, border_top:border_top+h, border_left:border_left+w])
        return res
