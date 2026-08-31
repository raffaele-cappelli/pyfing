import math
from pathlib import Path
import numpy as np
import cv2 as cv
from pyfing.definitions import Minutia
from ._nist_format import _read_nist_file


_CONVERSION_FACTOR = 1.0 / 2.54 / 2 # conversion to pixel coordinates at 500 dpi resolution

def _parse_minutia(values):
    if len(values) != 6:
        raise ValueError(f'Unexpected number of values in {values}')
    x = int(round(float(values[0]) * _CONVERSION_FACTOR))
    y = int(round(float(values[1]) * _CONVERSION_FACTOR))
    d = math.radians(int(values[2]) + math.pi) % math.tau
    t = values[3]
    q = 1.0
    # N.B. values[4] and values[5] are not considered, yet
    return Minutia(x, y, d, t, q)


def load_sd302_test_db(folder_path, include_limited_value = False, include_no_value = False, include_non_print = False):
    V_FLAGS = { "VALUE": True, "LIMITED": include_limited_value, "NOVALUE": include_no_value, "NONPRINT": include_non_print }
    db = []
    for file_path in Path(folder_path).rglob('*.lffs'):
        n = _read_nist_file(str(file_path))
        try:
            value = n.get('9.353')[0][0]
        except (TypeError, IndexError, AttributeError) as err:
            raise ValueError(f"Invalid or missing field '9.353'") from err
        if value not in V_FLAGS:
            raise ValueError(f"Unexpected assessment value: {value}")
        if not V_FLAGS[value]:
            continue

        image_array = np.frombuffer(n['13.999'], dtype=np.uint8)
        f = cv.imdecode(image_array, cv.IMREAD_GRAYSCALE)
        f = cv.resize(f, None, None, 0.5, 0.5, interpolation=cv.INTER_CUBIC) # resize to 500 dpi resolution
        if (n['13.009'], n['13.010']) != ([['1000']], [['1000']]):
            raise ValueError(f"Unexpected resolution: {(n['13.009'], n['13.010'])}")
        roi_field = n['9.300'][0]        
        if roi_field[2:4] != ['0', '0']:
            raise ValueError(f"Unexpected ROI origin: {roi_field}")
        if "9.331" not in n:
            minutiae = []
        else:
            minutiae = [_parse_minutia(x) for x in n["9.331"]]

        # Segmentation mask from ROI
        if len(roi_field) != 5:
            raise ValueError(f"Unexpected ROI format: {roi_field}")
        pts = np.array([[int(round(float(x) * _CONVERSION_FACTOR)), int(round(float(y) * _CONVERSION_FACTOR))] 
                        for x, y in (p.split(",") for p in roi_field[4].split("-"))], dtype=np.int32).reshape((-1, 1, 2))            
        x_min, y_min = pts.min(axis=(0, 1))
        mask = np.zeros_like(f)            
        cv.fillPoly(mask, [pts], 255)
        if "9.308" in n and n.get("9.309") in [ [['20', 'UNC']], [['20', 'UNC', '20', 'UNC']] ]:
            # Improve segmentation mask with quality mask (background/non-background blocks)
            block_size = 4 # 8 @ 1000 dpi, 4 @ 500 dpi
            gr = np.array([[255 if c != "0" else 0 for c in riga[0]] for riga in n['9.308']], dtype=np.uint8)
            h_f, w_f = f.shape[:2]
            mask1 = np.repeat(np.repeat(gr, block_size, axis=0), block_size, axis=1)
            h_m, w_m = mask1.shape
            mask_q = np.zeros_like(f)
            min_h, min_w = min(h_f-y_min, h_m), min(w_f-x_min, w_m)
            mask_q[y_min:y_min+min_h, x_min:x_min+min_w] = mask1[:min_h, :min_w]
            mask = cv.bitwise_and(mask, mask_q)

        db.append((f, mask, minutiae, file_path.name))
    return db