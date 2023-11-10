import cv2 as cv
import numpy as np
import IPython
import html
import base64
import math
import struct

def show(*images, enlarge_small_images = True, max_per_row = -1, font_size = 0):
    if len(images) == 2 and type(images[1])==str:
        images = [(images[0], images[1])]

    def convert(imgOrTuple):
        try:
            img, title = imgOrTuple
            if type(title)!=str:
                img, title = imgOrTuple, ''
        except ValueError:
            img, title = imgOrTuple, ''        
        if type(img)==str:
            data = img
        else:
            img = convert_for_display(img)
            if enlarge_small_images:
                REF_SCALE = 400
                h, w = img.shape[:2]
                if h<REF_SCALE or w<REF_SCALE:
                    scale = max(1, min(REF_SCALE//h, REF_SCALE//w))
                    img = cv.resize(img, (w*scale,h*scale), interpolation=cv.INTER_NEAREST)
            data = 'data:image/png;base64,' + base64.b64encode(cv.imencode('.png', img)[1]).decode('utf8')
        return data, title
    
    if max_per_row == -1:
        max_per_row = len(images)
    
    rows = [images[x:x+max_per_row] for x in range(0, len(images), max_per_row)]
    font = f"font-size: {font_size}px;" if font_size else ""
    
    html_content = ""
    for r in rows:
        l = [convert(t) for t in r]
        html_content += "".join(["<table><tr>"] 
                + [f"<td style='text-align:center;{font}'>{html.escape(t)}</td>" for _,t in l]    
                + ["</tr><tr>"] 
                + [f"<td style='text-align:center;'><img src='{d}'></td>" for d,_ in l]
                + ["</tr></table>"])
    IPython.display.display(IPython.display.HTML(html_content))

def convert_for_display(img):
    if img.dtype!=np.uint8:
        a, b = img.min(), img.max()
        if a==b:
            offset, mult, d = 0, 0, 1
        elif a<0:
            offset, mult, d = 128, 127, max(abs(a), abs(b))
        else:
            offset, mult, d = 0, 255, b
        img = np.clip(offset + mult*(img.astype(float))/d, 0, 255).astype(np.uint8)
    return img
        
def center_text(img, text, center, color, fontFace = cv.FONT_HERSHEY_PLAIN, fontScale = 1, thickness = 1, lineType = cv.LINE_AA, max_w = -1):
    while True:
        (w, h), _ = cv.getTextSize(text, fontFace, fontScale, thickness)
        if max_w<0 or w<max_w or fontScale<0.2:
            break
        fontScale *= 0.8
    pt = (center[0]-w//2, center[1]+h//2)
    cv.putText(img, text, pt, fontFace, fontScale, color, thickness, lineType)

    
def draw_hist(hist, height = 192, back_color = (160,225,240), border = 5):
    size = hist.size
    img = np.full((height, size+border*2, 3), back_color, dtype=np.uint8)
    nh = np.empty_like(hist, dtype=np.int32)
    cv.normalize(hist, nh, 0, height-1-border*2, cv.NORM_MINMAX, cv.CV_32S)
    for i in range(size):
        img[-border-nh[i]:-border,border+i,0:3] = i
    return img

_or_map = cv.applyColorMap(np.arange(255,-1,-1, dtype=np.uint8).reshape(1,256), cv.COLORMAP_JET).reshape(256,3)

def draw_orientations(fingerprint, orientations, strengths, mask, scale = 1, step = 8, border = 0):
    if strengths is None:
        strengths = np.ones_like(orientations)
    h, w = fingerprint.shape
    sf = cv.resize(fingerprint, (w*scale, h*scale), interpolation = cv.INTER_NEAREST)
    res = cv.cvtColor(sf, cv.COLOR_GRAY2BGR)
    d = (scale // 2) + 1
    sd = (step+1)//2
    #c = np.round(np.cos(orientations) * strengths * d * sd).astype(int)
    #s = np.round(-np.sin(orientations) * strengths * d * sd).astype(int) # minus for the direction of the y axis
    c = np.round(np.cos(orientations) * d * sd).astype(int)
    s = np.round(-np.sin(orientations) * d * sd).astype(int) # minus for the direction of the y axis
    colors = _or_map[np.clip(np.round(strengths * 255), 0, 255).astype(np.int32),:]
    thickness = 1 + scale // 5
    for y in range(border, h-border, step):
        for x in range(border, w-border, step):
            if mask is None or mask[y, x] != 0:
                ox, oy, color = c[y, x], s[y, x], tuple(colors[y, x].tolist())
                cv.line(res, (d+x*scale-ox,d+y*scale-oy), (d+x*scale+ox,d+y*scale+oy), color, thickness, cv.LINE_AA)
    return res


def draw_minutiae(fingerprint, minutiae, termination_color = (255,0,0), bifurcation_color = (0,0,255), other_color = (255,255,255)):
    res = cv.cvtColor(fingerprint, cv.COLOR_GRAY2BGR)
    for x, y, d, t in minutiae:
        color = termination_color if t=='T' else bifurcation_color if t=='B' else other_color
        if d is None:
            cv.drawMarker(res, (x,y), color, cv.MARKER_CROSS, 8)
        else:
            ox = int(round(math.cos(d) * 7))
            oy = int(round(math.sin(d) * 7))
            cv.circle(res, (int(x),int(y)), 3, color, 1, cv.LINE_AA)
            cv.line(res, (int(x),int(y)), (int(x)+ox,int(y)-oy), color, 1, cv.LINE_AA)
        
    return res


def biometric_performance(genuine_scores, impostor_scores):
    # scores: list of tuples (s, g) where s is a score and g indicates if it is a genuine (True) or imposter (False) score
    scores = [(s, True) for s in genuine_scores] + [(s, False) for s in impostor_scores]
    scores.sort(key=lambda x: x[0])

    max_w = len(genuine_scores) * len(impostor_scores)
    thr = 0.0
    weighted_fnm = 0
    weighted_fm = max_w
    # thr_info: list of tuples (s, weighted_fnm, weighted_fm, fnmr, fmr)
    thr_info = [(thr, weighted_fnm, weighted_fm, 0.0, 1.0)]
    for s, genuine in scores:
        if s != thr:
            thr_info.append( (thr, weighted_fnm, weighted_fm, weighted_fnm / max_w, weighted_fm / max_w) )
            thr = s
        if genuine: weighted_fnm += len(impostor_scores)
        else: weighted_fm -= len(genuine_scores)        
    thr_info.append( (1.0, max_w, 0, 1.0, 0.0) )

    thr_eer2_index = next(i for i, x in enumerate(thr_info) if x[1] >= x[2])
    thr_eer1 = thr_info[thr_eer2_index-1]
    thr_eer2 = thr_info[thr_eer2_index]
    thr_zero_fnmr = next(x for x in thr_info if x[1] > 0)
    thr_zero_fmr = next(x for x in thr_info if x[2] == 0)
    thr_fmr_100 = next(x for x in thr_info if x[4] <= 0.01)
    thr_fmr_1000 = next(x for x in thr_info if x[4] <= 0.001)
    thr_fmr_10000 = next(x for x in thr_info if x[4] <= 0.0001)

    eer = min((thr_eer1[3] + thr_eer1[4]) / 2, (thr_eer2[3] + thr_eer2[4]) / 2)
    zero_fnmr = thr_zero_fnmr[4]
    zero_fmr = thr_zero_fmr[3]
    fmr_100 = thr_fmr_100[3]
    fmr_1000 = thr_fmr_1000[3]
    fmr_10000 = thr_fmr_10000[3]
    return eer, zero_fnmr, zero_fmr, fmr_100, fmr_1000, fmr_10000


def unpack_minutiae_from_iso_template(buffer):
    """Unpack the minutiae in the first view of an ISO template. Returns a list of minutiae [(x,y,t,d)]"""

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

    minutia_types = {0:'O', 1: 'T', 2: 'B'}

    minutiae = []
    for x_t, y, d, q in struct.iter_unpack(">HHBB", buffer[28:28+minutiaeCount*6]):
        x = x_t & 0x3FFF # bits 0..13
        y &= 0x3FFF # bits 0..13
        t = minutia_types[x_t >> 14] # bits 14..15
        d = d * math.tau / 256 # from byte angle [0..255] to radians [0..360[
        d = math.pi - (math.pi - d) % math.tau # from [0..360[ to [-180..180[
        minutiae.append( (x, y, d, t) )

    return minutiae

