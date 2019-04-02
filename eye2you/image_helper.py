import numpy as np
import torch
import torchvision


def split_tensor_image(img, patch_size):
    '''Split a tensor into patches of (patch_size x patch_size).
    Pixels on the right get lost when patch_size does not match 
    
    Arguments:
        img {torch.Tensor} -- Tensor in CHW format
        patch_size {int} -- Edge length of the target patches
    
    Returns:
        torch.Tensor -- NCHW shaped tensor with all patches
    '''

    c, h, w = img.size()
    n_y = int(h / patch_size)
    n_x = int(w / patch_size)
    out = torch.zeros((n_x * n_y, c, patch_size, patch_size))
    for ii in range(n_x):
        for jj in range(n_y):
            out[ii * n_y +
                jj, :, :, :] = img[:, ii * patch_size:(ii + 1) * patch_size, jj * patch_size:(jj + 1) * patch_size]
    return out


def merge_tensor_image(patches):
    k, c, patch_size, _ = patches.size()
    n = int(np.sqrt(k))
    out = torch.zeros((c, patch_size * n, patch_size * n))
    for ii in range(n):
        for jj in range(n):
            out[:, ii * patch_size:(ii + 1) * patch_size, jj * patch_size:(jj + 1) * patch_size] = patches[ii * n +
                                                                                                           jj, :, :, :]
    return out


def merge_label_on_image(images, labels):
    out = torch.clamp(images + labels, 0, 1)
    return out