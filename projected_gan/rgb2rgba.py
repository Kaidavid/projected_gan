import PIL.Image
import numpy as np
import os
from pathlib import Path


def rgb2rgba_(im, alpha):
    '''
        Converts the RGB - 3channel images into RGBA - 4channel images
    '''
    assert 0 <= alpha <= 255
    
    ######-----------------   Incase of a numpy array or tensor   -------#####
    try:
        alpha_layer = np.full((im.shape[0], im.shape[1]), alpha)    # HWC
        return np.dstack((im, alpha_layer)).astype(np.uint8)

    ######-----------------   Incase of a PIL.Image   -----------------#####
    except:
        im.putalpha(alpha)
        return im.astype(np.uint8)
        

def save_rgba_(path, im, raw_idx):
    '''
        Save the rgba form of each image in a `pokemonrgba` folder
    '''
    assert isinstance(im, np.ndarray)
    os.makedirs(os.path.join(Path(path).parent, "pokemon_rgba"), exist_ok=True)
    
    image = PIL.Image.fromarray(im)
    image.save(os.path.join(Path(path).parent, "pokemon_rgba", raw_idx + ".png"))


    
def rgba2rgb_(im):
    '''
        Converts RGBA image to RGB image by removing the last transparency layer
        For pretraining with EfficientNet an RGB image is required
    '''
    ######-----------------   Incase of a PIL.Image   -----------------#####
    try:
        bg = PIL.Image.new("RGB", im.size, (255, 255, 255))
        return bg.paste(im, mask=im.split()[3])
    
    ######-----------------   Incase of a numpy array or tensor   -------#####
    except:
        return im[:, :3, :, :]



def padding_(im, desired_size):
    '''
        C, H, W = im.shape
        desired_size = [_, _]
        im must by a numpy array

        RGBA Images are truncated or padded white (255) to have identical resolution
        which is the desired_size 
    '''
    assert isinstance(im, np.ndarray)
    c, h, w = im.shape

    _h = np.abs(desired_size[0] - h)
    _w = np.abs(desired_size[1] - w)
    top, bottom, left, right = _h//2, _h - _h//2, _w - _w//2, _w//2
    
    ######-----------------   Truncating along height  -----------------#####
    if desired_size[0] < h:
        im = im[:, top:h - bottom, :]
    
    ######-----------------   Padding along height  --------------------#####
    else:
        top_pad = np.full((c, top, w), 255)
        bottom_pad = np.full((c, bottom, w), 255)
        
        im = np.concatenate((top_pad, im), axis=1)
        im = np.concatenate((im, bottom_pad), axis=1)

    _, h, _ = im.shape

    ######-----------------   Truncating along width  ------------------#####
    if desired_size[1] < w:
        im = im[:, :, left:w - right]

    ######-----------------   Padding along width  ---------------------#####
    else:
        left_pad = np.full((c, h, left), 255)
        right_pad = np.full((c, h, right), 255)
        
        im = np.concatenate((left_pad, im), axis=2)
        im = np.concatenate((im, right_pad), axis=2)       

    return im.astype(np.uint8)