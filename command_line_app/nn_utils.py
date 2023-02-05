import torch
import numpy as np
from PIL import Image
from constants import norm_mean, norm_std

def choose_device(gpu=False):
    if gpu and torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
    
    
def process_image(image):
    color_channel_mean = np.array(norm_mean)
    color_channel_std = np.array(norm_std)
    
    with Image.open(image) as img:
        if img.width > img.height:
            img.thumbnail((img.width, 256))
        else:
            img.thumbnail((256, img.height))
    
        img = img.crop((6, 6, 256, 256))
        np_img = np.array(img) / 255.0
            
        np_img = (np_img - color_channel_mean) / color_channel_std
        np_img = np_img.transpose((2,0,1))
        
        return torch.from_numpy(np_img) 
