"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """Converts a Tensor array into a numpy image array."""
    if not isinstance(input_image, np.ndarray):
        # PyTorch Tensor → numpy
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.detach()
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        # 1ch → 3ch
        if image_numpy.ndim == 3 and image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # C,H,W → H,W,C
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        # 画像の値域を判定して逆正規化
        min_val = float(image_numpy.min())
        if min_val < 0.0:
            # [-1,1] の場合は (x+1)/2 で 0〜1 にマッピング
            image_numpy = (image_numpy + 1.0) / 2.0
        # [0,1] 域に丸めて 0〜255 にスケーリング
        image_numpy = np.clip(image_numpy, 0.0, 1.0) * 255.0
    else:
        image_numpy = input_image
        
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
