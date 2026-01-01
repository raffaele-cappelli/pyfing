import numpy as np
import cv2 as cv


def draw_orientations(img, orientations, mask, scale = 1, step = 16, color = (255,0,0)):
    """Draws line segments, corresponding to the orientations, every step pixels over img, 
    only on mask pixels. The scale parameter allows to resize img. Returns the resulting image."""
    if img is None:
        img = np.full_like(orientations, 255, np.uint8)
    if mask  is None:
        mask = np.full_like(orientations, 255, np.uint8)
    h, w = orientations.shape
    img = cv.resize(img, (w*scale, h*scale), interpolation = cv.INTER_NEAREST)
    if len(img.shape) == 2: # assume it is a grayscale image: convert it to a color image
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    middle = scale // 2
    thickness = 1 + scale // 5
    hl = max(1, ((scale-1) / 2) + (step-1)*scale/2 - (thickness//2)) # half len of the segments
    iy, ix = np.ogrid[:h:step,:w:step] # coordinate of the points, reeady for indexing our matrices
    o = orientations[iy, ix]
    dx = np.round(np.cos(o) * hl).astype(np.int32)
    dy = np.round(-np.sin(o) * hl).astype(np.int32) # minus sin for the direction of the y axis    
    cx = middle + ix*scale # x coordinate of the center of each pixel
    cy = middle + iy*scale # y coordinate of the center of each pixel
    p1 = np.stack((cx - dx, cx + dx), 2) # first point of each line
    p2 = np.stack((cy - dy, cy + dy), 2) # second point of each line
    p = np.stack((p1, p2), 3)
    lines = p[mask[iy, ix]!=0,:,:] # select only foreground elements
    cv.polylines(img, lines, False, color, thickness, cv.LINE_AA)   
    return img


def draw_frequencies(fingerprint, periods, mask):
    """Draws a frequency map using the color scale reported in figure 2 of the paper. 
    The periods matrix must contain the inverse of the local frequency for each pixel."""
    if mask is None:
        mask = np.full_like(periods, 255, dtype=np.uint8)
    if fingerprint is None:
        fingerprint = mask
    periods = periods.clip(5, 20) - 3 # Range: [2..17]
    h = np.clip(periods*10, 0, 255).astype(np.uint8) # Range [20..170] (Hue is in [0..180])
    h[mask==0] = 0    
    s = np.full_like(h, 255)
    s[mask==0] = 0
    return cv.cvtColor(cv.merge((h, s, fingerprint)), cv.COLOR_HSV2BGR)


def draw_feature_map(intermediate_results, indices, feature_index_range, background = 0):
    """
    Creates a single image displaying a subset of intermediate outputs from CNN-based pyfing algorithms;
    each sub-image is normalized based on its own minimum and maximum values, scaled between 0 and 255.

    Parameters:
    -----------
        intermediate_results : List of tuples containing the intermediate outputs from CNN layers.
        indices : List of indices specifying which layers' outputs to visualize.
        feature_index_range : Tuple (start, stop) defining the range of feature indices to display.
        background : Pixel value for the background (default is 0, black).
    
    Returns:
    --------
        An image representing the combined feature map image, with pixel values in uint8 format.
    """
    h, w, n = intermediate_results[0][0].shape
    for i in indices:
        r, _ = intermediate_results[i]
        hr, wr, nr = r.shape
        if hr != h or wr != w:
            raise Exception("All feature must have the same size")
        n = max(n, nr)
    j1, j2 = feature_index_range
    n = min(j2 - j1, n)
    row_count = len(indices)
    map = np.full((row_count * h, n * w), background, np.uint8)
    for i in range(row_count):
        r, _ = intermediate_results[indices[i]]
        for j in range(j1, j2):
            if j < r.shape[2]:
                img = cv.normalize(r[..., j],  None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
                map[i*h:i*h+h, (j-j1)*w:(j-j1)*w+w] = img
    return map