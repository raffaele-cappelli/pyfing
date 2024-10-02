from .segmentation import *
from .orientations import *
from .frequencies import *
import numpy as np


_sufs_alg = None
_gmfs_alg = None
_snfoe_alg = None
_gbfoe_alg = None
_xsffe_alg = None
_snffe_alg = None


def segmentation(fingerprint, dpi = 500, method = "SUFS"):
    """
    Simple API for fingerprint segmentation.

    Parameters
    ----------
    fingerprint : a numpy array containing the fingerprint image (dtype: np.uint8).
    dpi : the fingerprint resolution.
    method : "SUFS" (requires Keras) or "GMFS".

    Returns
    ----------    
    A numpy array containing the segmentation mask, with the same size of the input fingerprint.
    """
    global _sufs_alg, _gmfs_alg
    if method == "SUFS":
        if _sufs_alg is None:
            _sufs_alg = Sufs()
        alg = _sufs_alg
    elif method == "GMFS":
        if _gmfs_alg is None:
            _gmfs_alg = Gmfs()
        alg = _gmfs_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    alg.parameters.image_dpi = dpi
    return alg.run(fingerprint)


def orientation_field_estimation(fingerprint, segmentation_mask = None, dpi = 500, method = "SNFOE"):
    """
    Simple API for fingerprint orientation field estimation.

    Parameters
    ----------
    fingerprint : a numpy array containing the fingerprint image (dtype: np.uint8).
    segmentation_mask : a numpy array containing the segmentation mask (dtype: np.uint8). If None, the whole image is taken.
    dpi : the fingerprint resolution.
    method : "SNFOE" (requires Keras) or "GBFOE".

    Returns
    ----------    
    A numpy array with the same size of the input fingerprint containing the orientation at each pixel, 
    in radians.
    """
    global _snfoe_alg, _gbfoe_alg
    if method == "SNFOE":
        if _snfoe_alg is None:
            _snfoe_alg = Snfoe()
        alg = _snfoe_alg
    elif method == "GBFOE":
        if _gbfoe_alg is None:
            _gbfoe_alg = Gbfoe()
        alg = _gbfoe_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return alg.run(fingerprint, segmentation_mask, dpi)[0]


def frequency_estimation(fingerprint, orientation_field, segmentation_mask = None, dpi = 500, method = "SNFFE"):
    """
    Simple API for fingerprint frequency estimation.

    Parameters
    ----------
    fingerprint : a numpy array containing the fingerprint image (dtype: np.uint8).
    orientation_field : a numpy array containing the ridge-line orientation (in radians) of each pixel (dtype: np.float32).
    segmentation_mask : a numpy array containing the segmentation mask (dtype: np.uint8). If None, the whole image is taken.
    dpi : the fingerprint resolution.
    method : "SNFFE" (requires Keras) or "XSFFE".

    Returns
    ----------    
    A numpy array with the same size of the input fingerprint containing the inverse of the frequency at each pixel.
    """
    global _xsffe_alg, _snffe_alg
    if method == "SNFFE":
        if _snffe_alg is None:
            _snffe_alg = Snffe()
        alg = _snffe_alg
    elif method == "XSFFE":
        if _xsffe_alg is None:
            _xsffe_alg = Xsffe()
        alg = _xsffe_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return alg.run(fingerprint, segmentation_mask, orientation_field, dpi)

