import math
import struct
from pyfing.definitions import Minutia


def load_minutiae_from_iso_template_file(path: str) -> list[Minutia]:
    """Reads the minutiae contained in the first view of an ISO template file. Returns a list of Minutia"""

    with open(path, mode='rb') as file:
        return _unpack_minutiae_from_iso_template(file.read())


_minutia_types = {0:'O', 1: 'E', 2: 'B'}

def _unpack_minutiae_from_iso_template(buffer):
    """Unpack the minutiae in the first view of an ISO template. Returns a list of Minutia"""

    # UInt32: formatIdentifier
    # UInt32: version
    # UInt32: length
    # UInt16: captureDeviceCertificationAndId
    # UInt16: imageSizeX
    # UInt16: imageSizeY
    # UInt16: resolutionX
    # UInt16: resolutionY
    # UInt8: viewCount
    # UInt8: reserved
    # -- For each view:
    #     UInt8: fingerPosition
    #     UInt8: viewNumber (bits 0..3), impressionType (bits 4..7)
    #     UInt8: fingerQuality
    #     UInt8: minutiaeCount
    #     -- For each minutia
    #         UInt16: x (bits 0..13), type (bits 14..15)
    #         UInt16: y (bits 0..13)
    #         UInt8: direction
    #         UInt8: quality

    (formatIdentifier, version, length, captureDeviceCertificationAndId, 
     imageSizeX, imageSizeY, resolutionX, resolutionY, viewCount, _) = struct.unpack(">IIIHHHHHBB", buffer[:24])

    if formatIdentifier != 0x464D5200:
        raise ValueError('Invalid ISO template format identifier.')

    if version != 0x20323000:
        raise ValueError('Unknown ISO template version.')

    # Unpack only the first view
    fingerPosition, viewNumber_ImpressionType, fingerQuality, minutiaeCount = struct.unpack(">BBBB", buffer[24:28])
    minutiae = []
    for x_t, y, d, q in struct.iter_unpack(">HHBB", buffer[28:28+minutiaeCount*6]):
        x = x_t & 0x3FFF # bits 0..13
        y &= 0x3FFF # bits 0..13
        t = _minutia_types[x_t >> 14] # bits 14..15
        d = d * math.tau / 256 # from byte angle [0..255] to radians [0..2*pi[        
        minutiae.append(Minutia(x, y, d, t, q))
    return minutiae
