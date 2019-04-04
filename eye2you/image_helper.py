import numpy as np
import torch
import torchvision
import cv2
from .transforms import SlidingWindowCrop


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


def split_tensor_image_sliding_window(img, patch_size, stride=1):
    if isinstance(patch_size, (list, tuple)):
        p_h, p_w = patch_size
    else:
        p_h = patch_size
        p_w = patch_size
    if isinstance(stride, (list, tuple)):
        s_h, s_w = stride
    else:
        s_h = stride
        s_w = stride
    c, h, w = img.shape

    n_h = int((h - p_h) / s_h) + 1
    n_w = int((w - p_w) / s_w) + 1
    patches = torch.zeros((n_h * n_w, c, p_h, p_w))
    for ii in range(n_h):
        for jj in range(n_w):
            patches[ii * n_w + jj, ...] = img[:, ii * s_h:ii * s_h + p_h, jj * s_w:jj * s_w + p_w]

    return patches


def split_tensor_image_sliding_window_generator(img, patch_size, stride=1):
    if isinstance(patch_size, (list, tuple)):
        p_h, p_w = patch_size
    else:
        p_h = patch_size
        p_w = patch_size
    if isinstance(stride, (list, tuple)):
        s_h, s_w = stride
    else:
        s_h = stride
        s_w = stride
    c, h, w = img.shape

    n_h = int((h - p_h) / s_h) + 1
    n_w = int((w - p_w) / s_w) + 1
    for ii in range(n_h):
        for jj in range(n_w):
            yield img[:, ii * s_h:ii * s_h + p_h, jj * s_w:jj * s_w + p_w]


def merge_labels(labels, img_size, stride=1, minimum_matches=0):
    if isinstance(img_size, (list, tuple)):
        h, w = img_size
    else:
        h = img_size
        w = img_size
    if isinstance(stride, (list, tuple)):
        s_h, s_w = stride
    else:
        s_h = stride
        s_w = stride

    if labels.dim() == 4:
        n, c, p_h, p_w = labels.shape
        if c != 1:
            raise ValueError('labels must be NHW or NCHW with C=1. This label is {0}'.format(labels.shape))
        labels = labels.squeeze()
    else:
        n, p_h, p_w = labels.shape
        c = 1

    n_h = int((h - p_h) / s_h) + 1
    n_w = int((w - p_w) / s_w) + 1
    img = torch.zeros((h, w)) - minimum_matches
    for ii in range(n_h):
        for jj in range(n_w):
            img[ii * s_h:ii * s_h + p_h, jj * s_w:jj * s_w + p_w] += labels[ii * n_w + n_h, :, :]
    img = torch.clamp(img, 0, 1)
    return img


def measure_iou(output, label):
    ious = []
    for ii in range(output.shape[0]):
        out = (output[ii, ...] > 0.5).detach().cpu().numpy()
        lab = (label[ii, ...] > 0.5).detach().cpu().numpy()
        union = out | lab
        intersect = out & lab
        iou = intersect.sum() / union.sum()
        if union.sum() == 0:
            iou = 0
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


def show_samples_from_loader(loader, steps=0):
    loader_iter = iter(loader)
    x, y = next(loader_iter)
    try:
        for ii in range(steps):
            x, y = next(loader_iter)
    except StopIteration:
        print('stopped after {0} steps, no data left'.format(ii))
    if y.shape[1] == 1:
        y = torch.cat((y, y, y), dim=1)
    elif y.shape[1] == 2:
        y = y[:, 1, :, :] > y[:, 0, :, :]
        y = torch.cat((y, y, y), dim=1)
    combined = torch.cat((x, y), dim=0)
    nrow = int(np.ceil(np.sqrt(x.shape[0])))
    grid = torchvision.utils.make_grid(x, nrow=nrow)
    grid2 = torchvision.utils.make_grid(y, nrow=nrow)
    grid = torch.cat((grid, grid2), dim=1)
    img = torchvision.transforms.ToPILImage()(grid)
    return img


def visualize_iou(output, target, colormap=((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1))):
    if output.shape[1] == 2:
        output = (output[:, (1,), :, :] > output[:, (0,), :, :]).float()
    if target.dim() == 3:
        target = target[:, None, :, :]
    if target.shape[1] == 2:
        target = target[:, (1,), :, :]

    correct = (output == target).float() * target
    missed = target * (1 - correct)
    false_detect = output * (1 - correct)
    nothing = 1 - torch.clamp(correct + missed + false_detect, 0, 1)
    areas = [nothing, correct, false_detect, missed]
    colormap = torch.Tensor(colormap)
    vis = sum([a * c.reshape(1, 3, 1, 1) for a, c in zip(areas, colormap)])

    nrow = int(np.ceil(np.sqrt(vis.shape[0])))
    grid = torchvision.utils.make_grid(vis, nrow=nrow)
    img = torchvision.transforms.ToPILImage()(grid)

    return img


def show_patches(patches):
    if patches.shape[1] not in (1, 3):
        patches = patches.view(-1, *patches.shape[2:])[:, None, :, :]
    nrow = int(np.ceil(np.sqrt(patches.shape[0])))
    grid = torchvision.utils.make_grid(patches, nrow=nrow)
    img = torchvision.transforms.ToPILImage()(grid)
    return img


def show_label(label):
    if label.dim() == 4 and label.shape[1] == 2:
        x = label[:, (1,), :, :] > label[:, (0,), :, :]
    elif label.dim() == 3:
        x = label[:, None, :, :]
    else:
        x = label
    nrow = int(np.ceil(np.sqrt(x.shape[0])))
    grid = torchvision.utils.make_grid(x, nrow=nrow)
    img = torchvision.transforms.ToPILImage()(grid)
    return img


def sliding_window_vessel(model, img, patch_size, stride=1, minimum_matches=0, device='cuda:0'):
    if isinstance(patch_size, (list, tuple)):
        p_h, p_w = patch_size
    else:
        p_h = patch_size
        p_w = patch_size
    if isinstance(stride, (list, tuple)):
        s_h, s_w = stride
    else:
        s_h = stride
        s_w = stride
    c, h, w = img.shape

    n_h = int((h - p_h) / s_h) + 1
    n_w = int((w - p_w) / s_w) + 1

    vessel = torch.zeros((1, h, w)) - minimum_matches
    with tqdm.tqdm(total=(n_h * n_w), desc='Vessel detection', unit='patch') as pb:
        for ii in range(n_h):
            for jj in range(n_w):
                x_cuda = img[:, ii * s_h:ii * s_h + p_h, jj * s_w:jj * s_w + p_w].to(device).unsqueeze(0)
                y_cuda = model(x_cuda)
                if y_cuda.shape[1] == 2:
                    label = (y_cuda[:, 1, :, :] > y_cuda[:, 0, :, :]).cpu().squeeze().float()
                else:
                    label = y_cuda.cpu().squeeze()
                vessel[0, ii * s_h:ii * s_h + p_h, jj * s_w:jj * s_w + p_w] += label
                pb.update(1)
    vessel = torch.clamp(vessel, 0, 1)
    return vessel


def denoise(img, ksize=(5, 5), morph=cv2.MORPH_RECT):
    kernel = cv2.getStructuringElement(morph, ksize)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img