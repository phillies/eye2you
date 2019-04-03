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


def measure_iou(output, label):
    ious = []
    for ii in range(output.shape[0]):
        out = (output[ii, ...] > 0.5).detach().cpu().numpy()
        lab = (label[ii, ...] > 0.5).detach().cpu().numpy()
        union = out | lab
        intersect = out & lab
        iou = intersect.sum() / union.sum()
        ious.append(iou)
    return ious


def sample_patches(images, labels, patch_size, number_patches):
    n, c, h, w = images.size()
    max_y = h - patch_size
    max_x = w - patch_size
    img_patch = torch.zeros((number_patches, c, patch_size, patch_size))
    lab_patch = torch.zeros((number_patches, 2, patch_size, patch_size))
    for ii in range(number_patches):
        img_index = np.random.randint(n)
        p_x = np.random.randint(max_x)
        p_y = np.random.randint(max_y)
        img_patch[ii, ...] = images[img_index, :, p_y:p_y + patch_size, p_x:p_x + patch_size]
        lab_patch[ii, 0, ...] = 1 - labels[img_index, :, p_y:p_y + patch_size, p_x:p_x + patch_size]
        lab_patch[ii, 1, ...] = labels[img_index, :, p_y:p_y + patch_size, p_x:p_x + patch_size]
    return img_patch, lab_patch


def parallel_variance(mean_a, count_a, var_a, mean_b, count_b, var_b):
    delta = mean_b - mean_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)
    M2 = m_a + m_b + delta**2 * count_a * count_b / (count_a + count_b)
    return M2 / (count_a + count_b - 1)