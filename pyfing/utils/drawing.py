import numpy as np
import cv2 as cv


def draw_orientations(img, orientations, mask, scale, step, color):
    """Draws line segments, corresponding to the orientations, every step pixels over img, 
    only on mask pixels. The scale parameter allows to resize img. Returns the resulting image."""
    h, w = mask.shape
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
