"""
Code based on: https://github.com/opencv/opencv/issues/20780
"""

import numpy as np

def intersect_rects(r1, r2):
    (r1x, r1y, r1w, r1h) = r1
    (r2x, r2y, r2w, r2h) = r2

    rx = max(r1x, r2x)
    rw = min(r1x+r1w, r2x+r2w) - rx

    ry = max(r1y, r2y)
    rh = min(r1y+r1h, r2y+r2h) - ry

    if rw > 0 and rh > 0:
        return np.array([rx, ry, rw, rh])
    else:
        return None

def rect_to_slice(rect):
    (rx, ry, rw, rh) = rect
    return np.s_[ry:ry+rh, rx:rx+rw]

def composite(src, dst=None, position=(0,0), background_color=None):
    # this is pronounced "com-PUH-sit" to all you non-native speakers

    (src_height, src_width, src_ch) = src.shape
    src_has_alpha = (src_ch == 4)

    # shortcut
    # composite() may be useful to use in imshow, so people "see" transparent data
    # and if data has no alpha channel, pass through
    if dst is None and not src_has_alpha:
        return src

    if dst is None:
        # background (grid or flat color)
        dst = np.empty((src_height, src_width, 3), dtype=np.uint8)
        if background_color is None:
            (i,j) = np.mgrid[0:src_height, 0:src_width]
            i = (i // 8) % 2
            j = (j // 8) % 2
            dst[i == j] = 192
            dst[i != j] = 255
        else:
            dst[:,:] = background_color

    (dst_height, dst_width) = dst.shape[:2]
    dst_has_alpha = (dst.shape[2] == 4)

    src_rect = np.array([0, 0, src_width, src_height])
    dst_rect = np.array([0, 0, dst_width, dst_height])
    offset = position + (0,0) # 4-tuple

    src_roi = intersect_rects(src_rect, dst_rect - offset)
    dst_roi = intersect_rects(dst_rect, src_rect + offset)

    if src_roi is not None: # there is overlap
        assert dst_roi is not None
        dst_slice = dst[rect_to_slice(dst_roi)]
        src_slice = src[rect_to_slice(src_roi)]

        if src_has_alpha:
            src_alpha = src_slice[:,:,3][..., None] # None adds a dimension for numpy broadcast rules
            if src_alpha.dtype == np.uint8:
                src_alpha = src_alpha / np.float32(255)

            blended = src_slice[:,:,0:3] * src_alpha + dst_slice[:,:,0:3] * (1-src_alpha)

        else:
            blended = src_slice[:,:,0:3]

        dst_slice[:,:,0:3] = blended.astype(dst.dtype)

        if dst_has_alpha:
            new_alpha = src_slice[:,:,:] + dst_slice[:,:,:] * (1-src_alpha)
            dst_slice[:,:,:] = new_alpha.astype(dst.dtype)

    return dst

def composite_from_bbox(src, dst=None, bbox=None, background_color=None):
    position = None
    if bbox:
        position = (int(dst.shape[1] * bbox[1]), int(dst.shape[0] * bbox[0]))
    return composite(src=src, dst=dst, position=position, background_color=background_color)


def noisy(noise_typ, image, var=0.1):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        # var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output