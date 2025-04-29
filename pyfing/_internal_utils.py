import keras
import warnings
import numpy as np
import cv2 as cv

def _predict_and_get_all_outputs(model, input):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # To suppress warning "The structure of `inputs` doesn't match the expected structure"
        # Creates a model to collect all intermediate outputs
        tmp_model = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        res = tmp_model(input, training=False)
        return [(r.numpy(), l) for r, l in zip(res, [l.name for l in model.layers])]


def _resize_and_crop_intermediate_output(w, h, border_left, border_top, border_right, border_bottom, output):
    w_full = w + border_left + border_right
    h_full = h + border_top + border_bottom
    *batch_size, oh, ow, n = output.shape
    if len(batch_size) != 0 and (len(batch_size) != 1 or batch_size[0] != 1):
        raise Exception("Invalid batch size")
    if ow != w_full or oh != h_full:
        imgs = [cv.resize(output[0,...,i], (w_full, h_full), interpolation=cv.INTER_NEAREST) for i in range(n)]
        output = np.empty((1, h_full, w_full, n), dtype=np.uint8)
        for i in range(n):
            output[0,...,i] = imgs[i]
    return output[0,border_top:border_top+h, border_left:border_left+w,:]

