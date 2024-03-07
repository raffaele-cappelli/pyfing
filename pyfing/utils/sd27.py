from glob import glob
import numpy as np
import cv2 as cv


_NIST_TAGS = {
    "1.001": ("Logical Record Length", [1], ""),
    "1.002": ("Version Number", [1], ""),
    "1.003": ("File Content", [2, 2], ""),
    "1.004": ("Type of Transaction", [1], ""),
    "1.005": ("Date", [1], ""),
    "1.006": ("Priority", [1], ""),
    "1.007": ("Destination Agency ID", [1], ""),
    "1.008": ("Originating Agency ID", [1], ""),
    "1.009": ("Transaction Control Number", [1], ""),
    "1.011": ("Native Scanning Resolution", [1], ""),
    "1.012": ("Nominal Transmitting Resolution", [1], "E"),
    "13.001": ("Logical Record Length", [1], ""),
    "13.002": ("Image Designation Character", [1], ""),
    "13.003": ("Impression Type", [1], ""),
    "13.004": ("Source Agency / ORI", [1], ""),
    "13.005": ("Latent Capture Date", [1], ""),
    "13.006": ("Horizontal Line Length", [1], ""),
    "13.007": ("Vertical Line Length", [1], ""),
    "13.008": ("Scale Units", [1], ""),
    "13.009": ("Horizontal Pixel Scale", [1], ""),
    "13.010": ("Vertical Pixel Scale", [1], ""),
    "13.011": ("Compression Algorithm", [1], ""),
    "13.012": ("Bits per Pixel", [1], ""),
    "13.013": ("Finger / Palm Position", [1], ""),
    "13.999": ("Image Data", [1], "EB"),    
    "14.001": ("Logical Record Length", [1], ""),
    "14.002": ("Image Designation Character", [1], ""),
    "14.003": ("Impression Type", [1], ""),
    "14.004": ("Source Agency / ORI", [1], ""),
    "14.005": ("Latent Capture Date", [1], ""),
    "14.006": ("Horizontal Line Length", [1], ""),
    "14.007": ("Vertical Line Length", [1], ""),
    "14.008": ("Scale Units", [1], ""),
    "14.009": ("Horizontal Pixel Scale", [1], ""),
    "14.010": ("Vertical Pixel Scale", [1], ""),
    "14.011": ("Compression Algorithm", [1], ""),
    "14.012": ("Bits per Pixel", [1], ""),
    "14.013": ("Finger / Palm Position", [1], ""),
    "14.999": ("Image Data", [1], "EB"),        
}

def _read_nist_file(path):
    def _get_sep(subfield_index, item_index, subfields, end_of_record):    
        if item_index == subfields[subfield_index] - 1:
            if subfield_index == len(subfields) - 1:
                return b'\x1C' if end_of_record else b'\x1D'
            else:
                return b'\x1E'
        else:
            return b'\x1F'

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
        tag_name, subfields, other = _NIST_TAGS[tag]
        end_of_record = "E" in other
        if "EB" in other: # binary field (must be at the end of record and of file!)
            sep = _get_sep(0, 0, [1], True)
            if content[-1:] != sep:
                raise Exception(f"Sep {sep} not found at the end of file")
            field = content[index:-1]
            index = len(content)
        else: # text field
            field = []
            for subfield_index in range(len(subfields)):
                subfield = []
                for item_index in range(subfields[subfield_index]):
                    sep = _get_sep(subfield_index, item_index, subfields, end_of_record)
                    end = content.find(sep, index)
                    if end == -1:
                        raise Exception(f"Sep {sep} not found starting from index {index}")
                    item = content[index:end].decode('ascii')
                    index = end + 1
                    subfield.append(item)
                field.append(subfield)
        fields[tag] = field
    return fields

    
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


def load_sd27_test_db(image_path, gt_path, db_name):    
    img_paths = glob(f"{image_path}/DATA/{db_name}/**/*L*.EFT", recursive=True)
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
        db.append((img, mask, gt_orientations, gt_mask, 500))
    return db

