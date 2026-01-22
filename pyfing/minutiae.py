import os
import keras
from keras import layers
import cv2 as cv
from abc import abstractmethod, ABC
from ._internal_utils import _predict_and_get_all_outputs, _resize_and_crop_intermediate_output
from .definitions import *


class EndToEndMinutiaExtractionParameters(Parameters):
    """
    Base class for the parameters of an end-to-end minutia extraction method.
    """
    pass


class EndToEndMinutiaExtractionAlgorithm(ABC):
    """
    Base class for end-to-end minutia extraction methods.
    """
    def __init__(self, parameters: EndToEndMinutiaExtractionParameters):
        self.parameters = parameters
    
    @abstractmethod
    def run(self, image: Image, dpi: int = 500, intermediate_results : list | None = None) -> list[Minutia]:
        raise NotImplementedError
    
    def run_on_db(self, images: list[Image], dpi_of_images: list[int]|None = None) -> list[list[Minutia]]:
        dpi_list = [500] * len(images) if dpi_of_images is None else dpi_of_images
        return [self.run(img, dpi) for img, dpi in zip(images, dpi_list)]

   
class LeaderParameters(EndToEndMinutiaExtractionParameters):
    """
    Parameters for the LEADER (Lightweight End-to-end Attention-gated Double skip-autoencodER) method.

    This class holds the configuration for fingerprint minutia extraction, including 
    image rescaling requirements, neural network input constraints, and detection thresholds.
    """
    def __init__(self, dnn_input_dpi = 500, dnn_input_size_multiple = 32, minutia_quality_threshold = 0.6, type_threshold = 0.5):
        """
        Initializes the LEADER parameters.

        Args:
            dnn_input_dpi (int): The expected resolution (DPI) for input images. 
                Images with different resolutions will be rescaled to match this value.
            dnn_input_size_multiple (int): The divisor required for input dimensions. 
                Input images are padded so that both width and height are multiples 
                of this value, as required by the underlying model architecture.
            min_minutia_score (float): The confidence threshold for minutia detection. 
                The default value (0.6) is tuned for a balanced precision/recall on 
                high-quality databases; different datasets may require adjustment 
                for optimal results.
            type_threshold (float): The decision threshold for classification. 
                Values above this threshold are typically classified as 'Terminations' (T), 
                while values below are classified as 'Bifurcations' (B).
        """
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple
        self.min_minutia_score = minutia_quality_threshold
        self.type_threshold = type_threshold


class Leader(EndToEndMinutiaExtractionAlgorithm):
    """
    Implementation of the LEADER (Lightweight End-to-end Attention-gated Double 
    skip-autoencodER) minutiae extraction method.

    This algorithm uses a dual skip-autoencoder architecture with attention-gate
    to perform end-to-end minutiae detection, orientation estimation, and type
    classification directly from fingerprint images.
    """
    def __init__(self, parameters : LeaderParameters | None = None, model_weights = None, model = None):
        """
        Initializes the LEADER algorithm.

        Args:
            parameters (LeaderParameters, optional): Configuration parameters for the algorithm. 
                If None, default LeaderParameters are used.
            model_weights (str, optional): Path to the .h5 or .weights.h5 file. If both 
                model_weights and model are None, the default internal model is loaded.
            model (keras.Model, optional): A pre-instantiated Keras model. If provided, 
                model_weights will be ignored.
        """
        if parameters is None:
            parameters = LeaderParameters()
        super().__init__(parameters)
        self.parameters = parameters
        if model_weights is None and model is None:
            model_weights = os.path.dirname(__file__) + "/models/LEADER.weights.h5"
        if model_weights is not None:
            self.model = Leader._build_model()
            self.model.load_weights(model_weights)
        elif model is not None:
            self.model = model


    _drop_out_rate = 0.02


    @staticmethod
    def _stem_block(filter_size, filter_count, max_pooling, dilation_rate, name, x):
        x1 = layers.Conv2D(filter_count, filter_size, padding="same", name=f"{name}_conv")(x)
        x2 = layers.Conv2D(filter_count, filter_size, dilation_rate=dilation_rate, padding="same", name=f"{name}_dilated_conv")(x)
        x = layers.Concatenate(name=f"{name}_conc")([x1, x2])
        x = layers.LayerNormalization(name=f"{name}_ln")(x)
        x = layers.Activation("gelu", name=f"{name}_activation")(x)
        if max_pooling:
            x = layers.MaxPooling2D(2, padding="same", name=f"{name}_mp")(x)
        else:
            x = layers.AveragePooling2D(2, padding="same", name=f"{name}_ap")(x)
        x = layers.SpatialDropout2D(Leader._drop_out_rate, name=f"{name}_drop_out")(x)
        return x


    @staticmethod
    def _sep_conv_block(filter_size, filter_count, name, x):
        x = layers.SeparableConv2D(filter_count, filter_size, padding="same", name=f"{name}_conv")(x)
        x = layers.LayerNormalization(name=f"{name}_ln")(x)
        return layers.Activation("gelu", name=f"{name}_activation")(x)
    

    @staticmethod
    def _inverse_bottleneck_conv_block(filter_size, final_filter_count, name, x):
        filter_count = x.shape[-1]
        x_input = x
        x = layers.DepthwiseConv2D(filter_size, padding="same", use_bias=False, name=f"{name}_depthwise_conv")(x) # N.B. with no bias
        x = layers.LayerNormalization(name=f"{name}_ln")(x)
        x = layers.Conv2D(filter_count * 4, 1, padding="same", activation = "gelu", name=f"{name}_conv_exp")(x)
        x = layers.Conv2D(filter_count, 1, padding="same", name=f"{name}_conv_red")(x)
        x = layers.Add(name=f"{name}_add")([x_input, x])
        return layers.Conv2D(final_filter_count, 1, padding="same", name=f"{name}_conv_adj")(x)
    

    @staticmethod
    def _downsample_block(name, x):
        c1 = x.shape[-1] // 2
        x1 = layers.MaxPooling2D(2, padding="same", name=f"{name}_mp")(x[...,:c1])
        x2 = layers.AveragePooling2D(2, padding="same", name=f"{name}_ap")(x[...,c1:])
        x = layers.Concatenate(name=f"{name}_conc")([x1, x2])
        x = layers.SpatialDropout2D(Leader._drop_out_rate, name=f"{name}_drop_out")(x)
        return x    


    @staticmethod
    def _upsample_block(name, x, skip_input = None):
        x = layers.UpSampling2D(2, name=f"{name}_up")(x)
        x = layers.SpatialDropout2D(Leader._drop_out_rate, name=f"{name}_drop_out")(x)
        return x if skip_input is None else layers.Concatenate(name=f"{name}_conc")([x, skip_input])


    @staticmethod
    def _head_block(filter_size, name, x):
        x = Leader._inverse_bottleneck_conv_block(filter_size, 6, f"{name}_ibc", x)
        x = Leader._upsample_block(f"{name}_up", x)
        x = layers.Conv2D(4, 1, padding="same", name=f"{name}_conv1")(x)
        x = layers.LayerNormalization(name=f"{name}_ln")(x)
        return layers.Conv2D(4, 1, padding="same", activation="gelu", name=f"{name}_conv2")(x)


    @staticmethod
    def _build_model():
        # Stem
        input = layers.Input((None, None, 1), name="input")
        x = Leader._stem_block(5, 8, True, 2, "stem0", input)
        x1 = Leader._stem_block(5, 16, False, 2, "stem1", input)

        # First Skip-autoencoder
        skip_inputs = []
        for i, fc in enumerate([16, 32, 64, 128]):
            x = Leader._sep_conv_block(5, fc, f"enc0_{i}", x)
            skip_inputs.append(x)
            x = Leader._downsample_block(f"enc0_{i}", x)            
        for i, fc in reversed(list(enumerate([16, 32, 64, 128]))):
            x = Leader._sep_conv_block(5, fc, f"dec0_{i}", x)
            x = Leader._upsample_block(f"dec0_{i}", x, skip_inputs[i])            

        # Attention-gate
        d = [layers.Conv2D(16, 3, dilation_rate=d, padding="same", activation="gelu", name=f"attention_conv_d{d}")(x) for d in (1, 3, 6)]
        xd = layers.Concatenate(name="attention_dconc")(d)
        xf = layers.Conv2D(32, 1, activation="sigmoid", name="attention_s")(xd)
        x = layers.Multiply(name="attention_mult")([x, xf])
        x = layers.Concatenate(name="attention_conc")([x, x1])

        # Second Skip-autoencoder
        skip_inputs = []
        for i, fc in enumerate([32, 64, 128, 32]):
            x = Leader._inverse_bottleneck_conv_block(7, fc, f"enc1_{i}", x)
            skip_inputs.append(x)
            x = Leader._downsample_block(f"enc1_{i}", x)
        for i, fc in reversed(list(enumerate([20, 42, 91, 27]))):
            x = Leader._inverse_bottleneck_conv_block(7, fc, f"dec1_{i}", x)
            x = Leader._upsample_block(f"dec1_{i}", x, skip_inputs[i])            

        # Head
        x = [Leader._head_block(7, f"head_{k}", x) for k in range(3)]
        pos = layers.Conv2D(1, 5, activation="sigmoid", padding="same", name="head_conv_pos")(x[0])
        dir = layers.Conv2D(2, 5, activation="linear", padding="same", name="head_conv_dir")(x[1])
        dir = layers.Lambda(lambda t: keras.ops.arctan2(t[...,1:2], t[...,0:1]), name="head_atan2")(dir)
        typ = layers.Conv2D(1, 5, activation="sigmoid", padding="same", name="head_conv_typ")(x[2])
        gw = cv.getGaussianKernel(5, 0).astype(np.float32)
        gw = np.outer(gw, gw)[..., np.newaxis, np.newaxis]
        gaussian_blur_layer = layers.Conv2D(filters=1, kernel_size=5, use_bias=False, padding='same', trainable=False, name="nms_Gaussian_blur")
        pos_gb = gaussian_blur_layer(pos)
        gaussian_blur_layer.set_weights([gw])
        pos_nms = layers.MaxPooling2D(pool_size=7, strides=1, padding="same", name="head_nms_mp")(pos_gb)
        pos_nms = layers.Lambda(lambda t: keras.ops.multiply(t[1], keras.ops.cast(t[0] == t[1], "float32")), name="head_nms")([pos_gb, pos_nms])
        out = layers.Concatenate(name="head_conc")([pos, dir, typ, pos_nms])
        return keras.Model(input, out)


    def _compute_single_size(self, x, size_mult, min_border):
        input_x = (x+size_mult-1) // size_mult * size_mult
        border_start = (input_x-x)//2
        border_end = input_x-x-border_start
        add_x = max(0, min_border - border_start) + max(0, min_border - border_end) + border_start + border_end
        return (x+add_x+size_mult-1) // size_mult * size_mult


    def _compute_size(self, w, h, size_mult, min_border):
        return self._compute_single_size(w, size_mult, min_border), self._compute_single_size(h, size_mult, min_border)


    def run(self, image: Image, dpi: int = 500, intermediate_results: list | None = None) -> list[Minutia]:
        """
        Extracts minutiae from a single fingerprint image.

        The image is automatically padded to satisfy the model's architectural constraints
        (multiples of dnn_input_size_multiple) and rescaled if the input DPI differs 
        from the expected resolution.

        Args:
            image (Image): The input grayscale fingerprint image (numpy array).
            dpi (int): The resolution of the input image. Defaults to 500.
            intermediate_results (list, optional): If a list is provided, it will be 
                populated with tuples of (output, layer_name) for each relevant 
                intermediate stage of the network.

        Returns:
            list[Minutia]: A list of detected Minutia objects.

        Raises:
            NotImplementedError: If intermediate_results are requested for images 
                requiring DPI scaling.
        """
        parameters = self.parameters
        if dpi != parameters.dnn_input_dpi:
            dpi_scale = parameters.dnn_input_dpi / dpi
            image = cv.resize(image, None, fx = dpi_scale, fy = dpi_scale, interpolation = cv.INTER_CUBIC)            
        h, w = image.shape    
        size_mult = parameters.dnn_input_size_multiple
        input_w, input_h = self._compute_size(w, h, size_mult, 0) 
        border_left, border_top = (input_w-w)//2, (input_h-h)//2
        border_right, border_bottom = input_w-w-border_left, input_h-h-border_top
        ir = cv.copyMakeBorder(image, border_top, border_bottom, border_left, border_right, cv.BORDER_CONSTANT, value = image[0,0].tolist())
        if intermediate_results is not None:
            if dpi != parameters.dnn_input_dpi: 
                raise NotImplementedError("Intermediate results are not available for this input resolution")
            res = _predict_and_get_all_outputs(self.model, np.dstack((ir,))[np.newaxis,...])
            intermediate_results += [(_resize_and_crop_intermediate_output(w, h, border_left, border_top, border_right, border_bottom, r), l) for r, l in res]
            out = res[-1][0][0]
        else:
            # From keras documentation: "For small numbers of inputs that fit in one batch, directly use __call__() for faster execution"
            out = self.model(ir[np.newaxis,..., np.newaxis], training = False).numpy()[0] 
        out = out[border_top:border_top+h, border_left:border_left+w]
        minutiae = self._get_minutiae(out)
        if dpi != parameters.dnn_input_dpi:
            scale = dpi / parameters.dnn_input_dpi
            minutiae = [Minutia(int(round(x*scale)), int(round((y*scale))), d, t, q) for x, y, d, t, q in minutiae]
        return minutiae


    def _get_minutiae(self, out) -> list[Minutia]:
        coords = np.argwhere(out[...,3] >= self.parameters.min_minutia_score).tolist()
        return [Minutia(ix, iy, out[iy, ix, 1].item(), 'E' if out[iy, ix, 2] >= self.parameters.type_threshold else 'B', out[iy, ix, 3].item()) for iy, ix in coords]  # P_NMS is used as minutia quality


    def run_on_db(self, images: list[Image], dpi_of_images = None, batch_size = 32, group_size = 32, verbose = False) -> list[list[Minutia]]:
        """
        Processes a list of images efficiently using batch prediction.

        Images are grouped by size to minimize padding and memory usage. Each group 
        is processed in batches to maximize GPU/CPU utilization.

        Args:
            images (list[Image]): A list of fingerprint images to process.
            dpi_of_images (list[int], optional): A list of DPI values corresponding 
                to each image. Currently, only the resolution defined in 
                parameters.dnn_input_dpi is supported for batch processing.
            batch_size (int): The number of images per prediction batch.
            group_size (int): The number of images to group together for size 
                homogenization (padding). Higher values may increase memory usage.
            verbose (bool): If True, prints progress and batch information to the console.

        Returns:
            list[list[Minutia]]: A list where each element is a list of detected 
                minutiae for the corresponding input image, preserving the original order.

        Raises:
            ValueError: If any image in the list has a DPI different from the 
                supported dnn_input_dpi.
        """
        size_mult = self.parameters.dnn_input_size_multiple
        if dpi_of_images is not None and any(dpi != self.parameters.dnn_input_dpi for dpi in dpi_of_images):
            raise ValueError(f"Only {self.parameters.dnn_input_dpi} DPI is supported")

        # Preserve the original order to return the results correctly, compute the padded height for each image
        indexed_data = list(enumerate(zip(images, [self._compute_single_size(x.shape[0], size_mult, 0) for x in images])))        
        indexed_data.sort(key=lambda x: (x[1][1], x[1][0].shape[1])) # Sort by padded height, then width
        original_indices = [x[0] for x in indexed_data]
        sorted_images = [x[1][0] for x in indexed_data]
        res = []
        for start_index in range(0, len(sorted_images), group_size):
            group_images = sorted_images[start_index:start_index+group_size]
            n = len(group_images)
            max_h, max_w = max(img.shape[0] for img in group_images), max(img.shape[1] for img in group_images)
            input_w, input_h = self._compute_size(max_w, max_h, size_mult, 0)
            net_input = np.empty((n, input_h, input_w, 1), np.uint8)
            border_info = []
            for k in range(n):
                image = sorted_images[start_index+k]
                h, w = image.shape
                border_left, border_top = (input_w-w)//2, (input_h-h)//2
                border_info.append((w, h, border_left, border_top))
                net_input[k,...,0] = cv.copyMakeBorder(image, border_top, input_h-h-border_top, border_left, input_w-w-border_left, cv.BORDER_CONSTANT, value=image[0,0].tolist())            
            if verbose:
                print(f"Start index: {start_index}, Group size: {group_size}, Input size: {input_w}x{input_h}, Current batch size: {batch_size}")
            batch_res = self.model.predict(net_input, batch_size, verbose = 0) # type: ignore
            for k in range(n):
                w, h, border_left, border_top = border_info[k]
                out = batch_res[k, border_top:border_top+h, border_left:border_left+w]
                res.append(self._get_minutiae(out))
            
        # Back to the original order before returning the results
        return [r for _, r in sorted(zip(original_indices, res), key=lambda x: x[0])]
    