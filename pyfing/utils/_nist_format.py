
_NIST_TAGS = {
    # Type-1 Transaction Information Record
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
    '1.013': ('Domain Name', ''),

    # Type-2 User-defined Descriptive Text Record
    '2.001': ('Logical Record Length', ''),
    '2.002': ('Image Designation Character', ''),
    '2.006': ('Send Copy To', ''),
    '2.010': ('Criminal Reference Number', ''),
    '2.011': ('Other Reference Number', ''),
    '2.034': ('Aliases', ''),
    '2.074': ('FGP', ''),

    # Type-9 Minutiae Data Record (legacy + EFS)
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

    # Type-9 Extended Feature Set (EFS)
    '9.300': ('Region of Interest', ''),
    '9.301': ('Orientation', ''),
    '9.302': ('Finger/Palm/Plantar Position', ''),
    '9.303': ('EFS Profile', ''), 
    '9.307': ('Pattern Classification', ''),
    '9.308': ('Ridge Quality/Confidence Map', ''),
    '9.309': ('Ridge Quality Map Format', ''), 
    '9.314': ('Tonal Reversal', ''),
    '9.315': ('Possible Lateral Reversal', ''),
    '9.320': ('Cores', ''),
    '9.321': ('Deltas', ''),
    '9.325': ('No Cores Present', ''),
    '9.326': ('No Deltas Present', ''),
    '9.327': ('No Distinctive Features Present', ''), 
    '9.331': ('Minutiae', ''),
    '9.334': ('No Minutiae Present', ''),
    '9.344': ('Dots', ''),
    '9.346': ('No Dots Present', ''),
    '9.347': ('No Incipient Ridges Present', ''),
    '9.348': ('No Creases or Linear Discontinuities Present', ''),
    '9.349': ('No Ridge Edge Features Present', ''),
    '9.350': ('EFS Method Of Feature Detection', ''),
    '9.353': ('EFS Examiner Analysis Assessment', ''),

    # Type-13 Variable-resolution Latent Image Record
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
    '13.996': ('Hash', ''),
    '13.999': ('Image Data', 'EB'),

    # Type-14 Variable-resolution Fingerprint Image Record
    '14.001': ('Logical Record Length', ''),
    '14.002': ('Image Designation Character', ''),
    '14.003': ('Impression Type', ''),
    '14.004': ('Source Agency / ORI', ''),
    '14.005': ('Capture Date', ''),
    '14.006': ('Horizontal Line Length', ''),
    '14.007': ('Vertical Line Length', ''),
    '14.008': ('Scale Units', ''),
    '14.009': ('Horizontal Pixel Scale', ''),
    '14.010': ('Vertical Pixel Scale', ''),
    '14.011': ('Compression Algorithm', ''),
    '14.012': ('Bits per Pixel', ''),
    '14.013': ('Finger / Palm Position', ''),
    '14.901': ('Friction ridge capture technology', ''),
    '14.996': ('Hash', ''),
    '14.999': ('Image Data', 'EB')
}


def _read_nist_file(path):
    _separators = 0x1C, 0x1D, 0x1E, 0x1F
    _sep_record, _sep_field, _sep_subfield, _sep_item = _separators
    _MAX_FIELD_LENGTH = 1000
    
    def _find_first_index_in_set(s, byte_set, start_index):
        for index, b in enumerate(s[start_index:start_index+_MAX_FIELD_LENGTH], start_index):
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

