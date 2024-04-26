from .segmentation import *
from .orientations import *
import numpy as np


_sufs_alg = None
_gmfs_alg = None
_snfoe_alg = None
_gbfoe_alg = None


def segmentation(fingerprint, dpi = 500, method = "SUFS"):
    """
    Simple API for fingerprint segmentation.

    Parameters
    ----------
    fingerprint : a numpy array containing the fingerprint image (dtype: np.uint8).
    dpi : the fingerprint resolution.
    method : SUFS (requires Keras) or GMFS.

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
    segmentation_mask : a numpy array containing the segmentation mask. If None, the whole image is taken.
    dpi : the fingerprint resolution.
    method : SNFOE (requires Keras) or GBFOE.

    Returns
    ----------    
    A numpy array with the same size of the input fingerprint containing the orientation at each pixel, 
    in radians.
    """
    global _snfoe_alg, _gbfoe_alg
    if method == "SNFOE":
        if _gbfoe_alg is None:
            _gbfoe_alg = Snfoe()
        alg = _gbfoe_alg
    elif method == "GBFOE":
        if _gbfoe_alg is None:
            _gbfoe_alg = Gbfoe()
        alg = _gbfoe_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return alg.run(fingerprint, segmentation_mask, dpi)[0]

