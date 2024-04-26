from abc import abstractmethod, ABC
import os
import math
import keras
import numpy as np
import cv2 as cv
from .definitions import Parameters, Image


class OrientationEstimationParameters(Parameters):
    """
    Base class for the parameters of an orientation estimation method.
    """
    pass


class OrientationEstimationAlgorithm(ABC):
    """
    Base class for orientation estimation methods.
    """
    def __init__(self, parameters: OrientationEstimationParameters):
        self.parameters = parameters
    
    @abstractmethod
    def run(self, image: Image, mask: Image = None, dpi: int = 500, intermediate_results = None) -> (np.ndarray, np.ndarray):
        raise NotImplementedError
    
    def run_on_db(self, images: [Image], masks: [Image], dpi_of_images: [int]) -> [(np.ndarray,np.ndarray)]:
        return [self.run(img, mask, dpi) for img, mask, dpi in zip(images, masks, dpi_of_images)]



def compute_orientation_RMSD(orientations, gt, gt_mask):
    """Returns the Root Mean Square Deviation (RMSD) of orientations with respect to gt, 
    computed only on gt_mask pixels and expressed in degrees"""
    diff = (orientations - gt + np.pi/2) % np.pi - np.pi/2
    return (np.sqrt((diff[gt_mask != 0]**2).mean()) * 180 / np.pi)    
  


class GbfoeParameters(OrientationEstimationParameters):
    """
    Parameters of GBFOE (Gradient-Based Fingerprint Orientation Estimation) method
    """

    def __init__(self, sigma_base = 27, sigma_multiplier = 43, sigma_smooth = 1.25, median_size = 5, percentile = 19):
        self.sigma_base = sigma_base
        self.sigma_multiplier = sigma_multiplier
        self.sigma_smooth = sigma_smooth
        self.median_size = median_size
        self.percentile = percentile        
        

class Gbfoe(OrientationEstimationAlgorithm):
    """
    Implementation of GBFOE (Gradient-Based Fingerprint Orientation Estimation) method.
    """

    def __init__(self, parameters : GbfoeParameters = None):
        if parameters is None:
            parameters = GbfoeParameters()
        super().__init__(parameters)
        self.parameters = parameters

    def run(self, image: Image, mask: Image = None, dpi: int = 500, intermediate_results = None) -> (np.ndarray,np.ndarray):        
        parameters = self.parameters

        if parameters.percentile > 0:
            masked_image = image[mask!=0] if mask is not None else image
            v1 = np.percentile(masked_image, parameters.percentile)
            v2 = np.percentile(masked_image, 100-parameters.percentile)
            if v2 > v1:
                image = np.clip((image.astype(np.int32)-v1)*255/(v2-v1), 0, 255).astype(np.uint8)
            if intermediate_results is not None: 
                intermediate_results.append((image, 'After contrast stretching'))        
                
        if parameters.sigma_smooth > 0:
            smooth_w = math.ceil(3*parameters.sigma_smooth)*2 + 1
            image = cv.GaussianBlur(image, (smooth_w, smooth_w), parameters.sigma_smooth)
            if intermediate_results is not None: 
                intermediate_results.append((image, 'After Gaussian blur'))

        if parameters.median_size > 1:
            image = cv.medianBlur(image, parameters.median_size)
            if intermediate_results is not None: 
                intermediate_results.append((image, 'After Median filter'))

        gx, gy = cv.Sobel(image, cv.CV_32F, 1, 0), cv.Sobel(image, cv.CV_32F, 0, 1)
        
        if mask is not None: # Do not use gradient information of background pixels
            gx[mask==0] = 0
            gy[mask==0] = 0

        gx2, gy2, g2xy = cv.pow(gx, 2), cv.pow(gy, 2), cv.multiply(gx, gy, scale = -2)
        
        # First calculate strengths with parameters.sigma_base
        strengths, *_ = self._compute_n_d_strengths(parameters.sigma_base, gx2, gy2, g2xy, mask)

        # Then calculate orientations and strengths with final sigma
        avg_s = strengths[mask!=0].mean()
        sigma = parameters.sigma_multiplier*(1-avg_s)
        strengths, n, d = self._compute_n_d_strengths(sigma, gx2, gy2, g2xy, mask)
        orientations = ((cv.phase(n, d) + np.pi) / 2) % np.pi
        return orientations, strengths
    
    def _compute_n_d_strengths(self, sigma, gx2, gy2, g2xy, mask):
        w = math.ceil(3*sigma)*2 + 1
        ksize = (w, w)
        sum_gx2, sum_gy2 = cv.GaussianBlur(gx2, ksize, 0), cv.GaussianBlur(gy2, ksize, 0)
        sum_gx2_gy2 = cv.add(sum_gx2, sum_gy2)
        n = cv.subtract(sum_gx2, sum_gy2)
        d = cv.GaussianBlur(g2xy, ksize, sigma)  # minus sign in gxy2 for orientation counter-clockwise
        strengths = np.divide(cv.sqrt(n**2 + d**2), sum_gx2_gy2, out=np.zeros_like(gx2), where=sum_gx2_gy2!=0)
        if mask is not None:
            strengths[mask==0] = 0
        return strengths, n, d
    

    
class SnfoeParameters(OrientationEstimationParameters):
    """
    Parameters of SNFOE (Simple Network for Fingerprint Orientation Estimation) method.
    """
    def __init__(self, dnn_input_dpi = 500, dnn_input_size_multiple = 32):
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple


class Snfoe(OrientationEstimationAlgorithm):
    """
    Implementation of SNFOE (Simple Network for Fingerprint Orientation Estimation) method.
    If both model_weights and model are None, the default model installed with the package is loaded.
    """
    
    def __init__(self, parameters : SnfoeParameters = None, model_weights = None, model = None):
        if parameters is None:
            parameters = SnfoeParameters()
        super().__init__(parameters)
        self.parameters = parameters
        if model_weights is None and model is None:
            model_weights = os.path.dirname(__file__) + "/models/SNFOE.weights.h5"
        if model_weights is not None:
            self.model = self._build_model()
            self.model.load_weights(model_weights)
        elif model is not None:
            self.model = model
    

    def _build_model(self):
        layers = keras.layers
        input = layers.Input((None, None, 2), name="input")
        x = input

        # Encoder    
        intermediate_outputs = []
        for i in range(5):
            x = layers.Conv2D(16*2**i, 5, padding="same", activation = "relu", name=f"enc_{i}_conv_5")(x)
            x = layers.BatchNormalization(name=f"enc_{i}_bn")(x)
            intermediate_outputs.append(x)
            x = layers.MaxPooling2D(2, padding="same", name=f"enc_{i}_dn_maxpool_2")(x)

        # Decoder
        for i, intermediate_output in enumerate(reversed(intermediate_outputs)):
            x = layers.Conv2D(16*2**(4-i), 5, padding="same", activation = "relu", name=f"dec_{i}_conv_5")(x)
            x = layers.BatchNormalization(name=f"dec_{i}_bn")(x)
            x = layers.UpSampling2D(2, name=f"dec_{i}_up_2")(x)
            x = layers.Concatenate(name=f"dec_{i}_skip_connection")([x, intermediate_output])    

        # Head
        x = layers.Concatenate(name="concat_mask")([x, input[...,1:]])
        x = layers.Conv2D(16, 5, padding="same", activation="relu", name="head_0_conv_5")(x)
        x = layers.BatchNormalization(name="head_0_bn")(x)    
        x = layers.Conv2D(2, 3, activation="linear", padding="same", name="head_linear_conv_3")(x)
        x = layers.Lambda(lambda t: keras.ops.arctan2(t[...,1], t[...,0])/2, name="head_atan2")(x)

        return keras.Model(input, x)        


    def run(self, image: Image, mask: Image = None, dpi: int = 500, intermediate_results = None) -> (np.ndarray,np.ndarray):        
        parameters = self.parameters
        original_image_h, original_image_w = image.shape

        if dpi != parameters.dnn_input_dpi:
            f = parameters.dnn_input_dpi / dpi
            image = cv.resize(image, None, fx = f, fy = f, interpolation = cv.INTER_CUBIC)
            mask =  cv.resize(mask, None, fx = f, fy = f, interpolation = cv.INTER_NEAREST)

        h, w = image.shape    
        size_mult = parameters.dnn_input_size_multiple
        input_w, input_h = (w+size_mult-1)//size_mult*size_mult, (h+size_mult-1)//size_mult*size_mult
        border_left, border_top = (input_w-w)//2, (input_h-h)//2
        ir = cv.copyMakeBorder(image, border_top, input_h-h-border_top, border_left, input_w-w-border_left, cv.BORDER_CONSTANT, value = image[0,0].tolist())
        mr = cv.copyMakeBorder(mask//255, border_top, input_h-h-border_top, border_left, input_w-w-border_left, cv.BORDER_CONSTANT)
        orientations = self.model(np.dstack((ir, mr))[np.newaxis,...], training = False).numpy()[0] # From keras documentation: "For small numbers of inputs that fit in one batch, directly use __call__() for faster execution"
        orientations = orientations[border_top:border_top+h, border_left:border_left+w]

        if dpi != parameters.dnn_input_dpi:
            cos2, sin2 = np.cos(orientations*2), np.sin(orientations*2)
            cos2 = cv.resize(cos2, (original_image_w, original_image_h), interpolation = cv.INTER_CUBIC)
            sin2 = cv.resize(sin2, (original_image_w, original_image_h), interpolation = cv.INTER_CUBIC)        
            orientations = np.arctan2(sin2, cos2)/2            

        strengths = np.ones_like(orientations) # no strength (coherence) information is produced by this method
        return orientations, strengths
