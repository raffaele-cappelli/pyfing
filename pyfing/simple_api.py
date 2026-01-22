from .segmentation import *
from .orientations import *
from .frequencies import *
from .enhancement import *
from .minutiae import *
import numpy as np


_sufs_alg = None
_gmfs_alg = None
_snfoe_alg = None
_gbfoe_alg = None
_xsffe_alg = None
_snffe_alg = None
_snfen_alg = None
_gbfen_alg = None
_leader_alg = None

def fingerprint_segmentation(fingerprint: Image, dpi: int = 500, method: str = "SUFS") -> Image:
    """
    Simple API for fingerprint segmentation.

    Parameters
    ----------
    fingerprint : a numpy array containing the fingerprint image (dtype: np.uint8).
    dpi : the fingerprint resolution.
    method : "SUFS" (requires Keras) or "GMFS".

    Returns
    ----------    
    A numpy array containing the segmentation mask, with the same shape of the input fingerprint.
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


def orientation_field_estimation(fingerprint: Image, segmentation_mask: Image|None = None, dpi: int = 500, method: str = "SNFOE") -> np.ndarray:
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
    A numpy array with the same shape of the input fingerprint containing the orientation at each pixel, 
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


def frequency_estimation(fingerprint: Image, orientation_field: np.ndarray, segmentation_mask: Image | None = None, dpi: int = 500, method: str = "SNFFE") -> np.ndarray:
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
    A numpy array with the same shape of the input fingerprint containing the inverse of the frequency at each pixel.
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


def fingerprint_enhancement(fingerprint: Image, orientation_field: np.ndarray, ridge_period_map: np.ndarray, segmentation_mask: Image | None = None, 
                            dpi: int = 500, method: str = "SNFEN") -> Image:
    """
    Simple API for fingerprint enhancement.

    Parameters
    ----------
    fingerprint : a numpy array containing the fingerprint image (dtype: np.uint8).
    orientation_field : a numpy array containing the ridge-line orientation (in radians) of each pixel (dtype: np.float32).
    ridge_period_map : a numpy array containing the ridge-line period of each pixel (the inverse of the frequency).
    segmentation_mask : a numpy array containing the segmentation mask (dtype: np.uint8). If None, the whole image is taken.
    dpi : the fingerprint resolution.
    method : "SNFEN" (requires Keras) or "GBFEN".

    Returns
    ----------    
    The enhanced image, a nearly-binary image, with ridge-line pixels appearing near-white and valleys near-black.
    """
    global _snfen_alg, _gbfen_alg
    if method == "SNFEN":
        if _snfen_alg is None:
            _snfen_alg = Snfen()
        alg = _snfen_alg
    elif method == "GBFEN":
        if _gbfen_alg is None:
            _gbfen_alg = Gbfen()
        alg = _gbfen_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    if segmentation_mask is None:
        segmentation_mask = np.full_like(fingerprint, 255)
    return alg.run(fingerprint, segmentation_mask, orientation_field, ridge_period_map, dpi)


def minutiae_extraction(fingerprint: Image, dpi: int = 500, method: str = "LEADER") -> list[Minutia]:
    """
    Simple API for fingerprint minutiae extraction.

    Parameters
    ----------
    fingerprint : Image
        A numpy array containing the fingerprint image (dtype: np.uint8). 
    dpi : int, optional
        The fingerprint resolution in dots per inch. Default is 500.
    method : str, optional
        The extraction algorithm to use. Currently only "LEADER" is supported.

    Returns
    -------
    list[Minutia]
        A list of detected minutiae.
    """
    global _leader_alg
    if method == "LEADER":
        if _leader_alg is None:
            _leader_alg = Leader()
        alg = _leader_alg
    else:
        raise ValueError(f"Invalid method ({method})")
    return alg.run(fingerprint, dpi)

