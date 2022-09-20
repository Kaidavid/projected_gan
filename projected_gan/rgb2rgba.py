import PIL.Image
import numpy as np

def rgb2rgba_(im, alpha):
    assert 0 <= alpha <= 255
    try:
        alpha_layer = np.full((im.shape[0], im.shape[1]), alpha)    # HWC
        return np.dstack((im, alpha_layer))
    except:
        im.putalpha(alpha)
        return im
        

def rgba2rgb_(im):      # For pretraining with EfficientNet
    try:        # Incase of a PIL.Image
        bg = PIL.Image.new("RGB", im.size, (255, 255, 255))
        return bg.paste(im, mask=im.split()[3])
    except:     # Incase of numpy array or tensor
        return im[:, :3, :, :]


def padding_(im, desired_size):     #CHW
    assert isinstance(im, np.ndarray)
    c, h, w = im.shape
    _h = desired_size[0] - h
    _w = desired_size[1] - w
    top, bottom, left, right = _h//2, _h - _h//2, _w - _w//2, _w//2
    
    #   TRUNCATE
    if _h < 0:
        im = im[:, -top:desired_size[0] - bottom, :]
    if _w < 0:
        im = im[:, :, -left:desired_size[1] - right]

    #   PAD with white background
    else:
        top_pad = np.full((c, top, w), 255)
        bottom_pad = np.full((c, bottom, w), 255)
        im = np.concatenate((top_pad, im), axis=1)
        im = np.concatenate((im, bottom_pad), axis=1)
        
        _, h, w = im.shape
        left_pad = np.full((c, h, left), 255)
        right_pad = np.full((c, h, right), 255)
        
        im = np.concatenate((left_pad, im), axis=2)
        im = np.concatenate((im, right_pad), axis=2)       

    return im.astype(np.uint8)