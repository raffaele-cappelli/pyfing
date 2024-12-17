import math
from glob import glob
import numpy as np
import cv2 as cv


_NIST_TAGS = {
    '1.001': ('Logical Record Length', ''),    
    '1.002': ('Version Number', ''),
    '1.003': ('File Content', ''),
    '1.004': ('Type of Transaction', ''),
    '1.005': ('Date', ''),
    '1.006': ('Priority', ''),
    '1.007': ('Destination Agency ID', ''),
    '1.008': ('Originating Agency ID', ''),
    '1.009': ('Transaction Control Number', ''),
    '1.011': ('Native Scanning Resolution', ''),
    '1.012': ('Nominal Transmitting Resolution', ''),
    '9.001': ('Logical Record Length', ''),
    '9.002': ('Image Designation Character', ''),
    '9.003': ('Impression Type', ''),
    '9.004': ('Minutiae Format', ''),
    '9.005': ('Originating Fingerprint Reading System', ''),
    '9.006': ('Finger Position', ''),
    '9.007': ('Fingerprint Pattern Classification(s)', ''),
    '9.008': ('Core(s) Position', ''),
    '9.009': ('Delta(s) Position', ''),
    '9.010': ('Number of Minutiae', ''),
    '9.011': ('Minutiae Ridge Count Indicator', ''),
    '9.012': ('Minutiae and Ridge Count Data', ''),
    '9.014': ('Finger Number', ''),
    '9.015': ('Number of Minutiae', ''),
    '9.016': ('Fingerprint Characterization Process', ''),
    '9.017': ('AFIS/FBI Pattern Classification', ''),
    '9.020': ('Orientation Uncertainty', ''),
    '9.021': ('Core Attributes', ''),
    '9.022': ('Delta Attributes', ''),
    '9.023': ('Minutiae Attributes', ''),
    '13.001': ('Logical Record Length', ''),
    '13.002': ('Image Designation Character', ''),
    '13.003': ('Impression Type', ''),
    '13.004': ('Source Agency / ORI', ''),
    '13.005': ('Latent Capture Date', ''),
    '13.006': ('Horizontal Line Length', ''),
    '13.007': ('Vertical Line Length', ''),
    '13.008': ('Scale Units', ''),
    '13.009': ('Horizontal Pixel Scale', ''),
    '13.010': ('Vertical Pixel Scale', ''),
    '13.011': ('Compression Algorithm', ''),
    '13.012': ('Bits per Pixel', ''),
    '13.013': ('Finger / Palm Position', ''),
    '13.999': ('Image Data', 'EB'),
    '14.001': ('Logical Record Length', ''),
    '14.002': ('Image Designation Character', ''),
    '14.003': ('Impression Type', ''),
    '14.004': ('Source Agency / ORI', ''),
    '14.005': ('Latent Capture Date', ''),
    '14.006': ('Horizontal Line Length', ''),
    '14.007': ('Vertical Line Length', ''),
    '14.008': ('Scale Units', ''),
    '14.009': ('Horizontal Pixel Scale', ''),
    '14.010': ('Vertical Pixel Scale', ''),
    '14.011': ('Compression Algorithm', ''),
    '14.012': ('Bits per Pixel', ''),
    '14.013': ('Finger / Palm Position', ''),
    '14.999': ('Image Data', 'EB')
}


def _read_nist_file(path):
    _separators = 0x1C, 0x1D, 0x1E, 0x1F
    _sep_record, _sep_field, _sep_subfield, _sep_item = _separators
    
    def _find_first_index_in_set(s, byte_set, start_index):
        for index, b in enumerate(s[start_index:start_index+30], start_index):
            if b in byte_set:
                return index
        return -1
    
    with open(path, 'rb') as f:
        content = f.read()
    fields = {}
    index = 0
    while index < len(content):
        end_of_tag = content.find(b':', index)
        if end_of_tag == -1:
            raise Exception(f"End of tag not found starting from index {index}")
        tag = content[index:end_of_tag].decode("ascii")
        index = end_of_tag + 1
        tag_name, field_type = _NIST_TAGS[tag]
        if "EB" in field_type: # binary field (must be at the end of record and of file!)
            if content[-1] != _sep_record:
                raise Exception(f"Sep {_sep_record} not found at the end of file")
            field = content[index:-1]
            index = len(content)
        else: # text field
            field = []
            current_items = []
            sep = None
            while sep not in [_sep_record, _sep_field]:
                sep_index =  _find_first_index_in_set(content, _separators, index)
                if sep_index == -1:
                    raise Exception(f"Separator not found from index {index}")
                sep = content[sep_index]
                item = content[index:sep_index].decode('ascii')
                index = sep_index + 1
                current_items.append(item)
                if sep != _sep_item: # end of the current subfield
                    field.append(current_items)
                    current_items = []
        fields[tag] = field
    return fields


def _parse_minutia(xyd, q, t):
    x = int(round(float(xyd[0:4])*19.69/100))
    y = int(round((3900-float(xyd[4:8]))*19.69/100))
    d = (math.radians(int(xyd[8:11])) + math.pi) % (2*math.pi)
    q = int(q)
    t = 'T' if t == 'A' else 'B' if t == 'B' else 'O'
    return x, y, d, t, q


def _load_sd27_orientations(path, border):
    shape = (48*16, 50*16)
    values = np.loadtxt(path, np.float32, delimiter=',')
    orientations = np.zeros(shape, np.float32)
    mask = np.zeros(shape, np.uint8)
    for i in range(48):
        for j in range(50):
            m = 255 if values[i, j]!=91 else 0
            mask[i*16+border, j*16+border] = m
            if m != 0:
                orientations[i*16+border, j*16+border] = values[i, j] * np.pi / 180  
    return mask, orientations    


def _create_pixelwise_foreground(fg, s):
    """Creates a segmentation mask starting from ground truth foreground values"""
    return cv.morphologyEx(fg, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (s, s)))


def load_sd27_test_db(image_path, gt_path, db_name, include_orientations_and_dpi = False, include_minutiae_and_name = False):
    img_paths = sorted(glob(f"{image_path}/DATA/{db_name}/**/*L*.EFT", recursive=True))
    border_gt = 8
    db = []
    for index in range(len(img_paths)):
        fields = _read_nist_file(img_paths[index])
        w_input = int(fields["13.006"][0][0])
        h_input = int(fields["13.007"][0][0])
        img = np.frombuffer(fields["13.999"], np.uint8).reshape(h_input, w_input)
        img_name = img_paths[index][-11:-7]
        gt_mask, gt_orientations = _load_sd27_orientations(f"{gt_path}/OF_manual/{img_name}.txt", border_gt)
        mask = _create_pixelwise_foreground(gt_mask, 16)
        fields_min = _read_nist_file(img_paths[index][:-5]+"I.LFF")
        minutiae = [_parse_minutia(xyd, q, t) for _, xyd, q, t in fields_min['9.012']]
        t = [img, mask]
        if include_orientations_and_dpi:
            t += [gt_orientations, gt_mask, 500]
        if include_minutiae_and_name:
            t += [minutiae, img_paths[index][-11:-4]]
        db.append(tuple(t))
    return db
