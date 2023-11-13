import cv2 as cv
import numpy as np
import IPython
import html
import base64
import math

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
            img = _convert_for_display(img)
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

def _convert_for_display(img):
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
