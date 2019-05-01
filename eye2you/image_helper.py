import numpy as np
import torch
import torchvision
import cv2
import tqdm
from PIL import Image


def float_to_uint8(img, min_val=None, max_val=None):
    if min_val is None:
        min_val = img.min()
    if max_val is None:
        max_val = img.max()
    img_scale = img - min_val
    img_scale = img_scale / (max_val - min_val)
    img_scale = (img_scale * 255).astype(np.ubyte)
    return img_scale


def PIL_to_cv2(img):
    '''Converts PIL uint8 image into numpy array. No conversion applied.

    Arguments:
        img {PIL.Image} -- PIL image object

    Returns:
        numpy.array -- numpy array with same data as PIL image
    '''

    return np.array(img)


def PIL_to_torch(img):
    '''Converts PIL uint8 image into torch tensor. No conversion applied.

    Arguments:
        img {PIL.Image} -- PIL image object

    Returns:
        torch.Tensor -- torch tensor with same data as PIL image
    '''

    return torchvision.transforms.ToTensor()(img)


def torch_to_PIL(img):
    '''Converts torch tensor image into PIL uint8. No conversion applied.

    Arguments:
        img {torch.tensor} -- torch tensor

    Returns:
        PIL.Image -- PIL Image with same data as torch tensor
    '''

    return torchvision.transforms.ToPILImage()(img)


def torch_to_cv2(img):
    '''Converts torch format (NCHW) to cv2 format (HWC)

    Arguments:
        img {torch.Tensor} -- NCHW shaped with n=1 or CHW shaped torch tensor

    Returns:
        [numpy.ndarray] -- PIL image in uint8 format
    '''
    img = np.transpose(img.squeeze().numpy(), axes=(1, 2, 0))
    return img


def cv2_to_PIL(img, min_val=None, max_val=None):
    '''Converts the cv2 image or numpy array of arbitrary scale to a PIL image with
    uint8 format. Upper and lower bound for scaling can be given, e.g. 0.0 and 1.0, otherwise
    min and max values of image are used for 0 and 255. No clipping is applied. Passing a lower
    bound larger than the smallest value in the image can lead to values <0 and undefined behaviour
    in the conversion.

    Arguments:
        img {numpy.array} -- Numpy array compatible to PIL, e.g. h,w,1 or h,w,3 shaped

    Keyword Arguments:
        min_val {float} -- lower bound for scaling (default: {None})
        max_val {float} -- upper bound for scaling (default: {None})

    Returns:
        [PIL.Image] -- PIL image in uint8 format
    '''

    img_scale = float_to_uint8(img, min_val, max_val)
    pil_img = Image.fromarray(img_scale)
    return pil_img


def cv2_to_torch(img):
    '''Converts cv2 format (HWC) to torch format (NCHW)

    Arguments:
        img {numpy.array} -- Numpy array compatible to PIL, e.g. h,w,1 or h,w,3 shaped

    Keyword Arguments:
        min_val {float} -- lower bound for scaling (default: {None})
        max_val {float} -- upper bound for scaling (default: {None})

    Returns:
        [PIL.Image] -- PIL image in uint8 format
    '''
    img = np.transpose(img, axes=(1, 2, 0))
    torch_img = torch.Tensor(img).unsqueeze(0)
    return torch_img


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
            out[ii * n_y + jj, :, :, :] = img[:, ii * patch_size:(ii + 1) * patch_size, jj * patch_size:(jj + 1) *
                                              patch_size]
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


def loader_to_images(loader, prefix=None, max_counter=None):
    counter = 0
    if max_counter == None:
        max_counter = len(loader)
    for source, _ in loader:
        if isinstance(source, (list, tuple)):
            source = source[0]

        nrow = int(np.ceil(np.sqrt(source.shape[0])))
        grid = torchvision.utils.make_grid(source, nrow=nrow)
        img = torchvision.transforms.ToPILImage()(grid)
        img.save(prefix + f'{counter:04d}.png')
        counter += 1
        if counter >= max_counter:
            break


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


def find_retina_boxes(im,
                      display=False,
                      dp=1.0,
                      param1=60,
                      param2=50,
                      minimum_circle_distance=0.2,
                      minimum_radius=0.4,
                      maximum_radius=0.65,
                      max_patch_size=800,
                      max_distance_center=0.05,
                      param1_limit=40,
                      param1_step=10,
                      param2_limit=20,
                      param2_step=10):
    '''Finds the inner and outer box around the retina using openCV
    HoughCircles. If more than one circle is found returns the circle with the center
    closest to the image center.
    Recursively decreases first param1 then param2 if no circles are found until the limit for both parameter is reached.
    Be cautious, setting the parameter limits too low will lead to very high computation time.
    For details see OpenCV documentation [https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d]

    Arguments:
        im {numpy.array} -- HWC-shaped uint8 numpy array containing the image data

    Keyword Arguments:
        display {bool} -- If true returns image with all circles drawn on it as 5th entry in tuple  (default: {False})
        dp {float} -- Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1, the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height. (default: {1.0})
        param1 {int} -- the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller). (default: {60})
        param2 {int} -- the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first. (default: {50})
        minimum_circle_distance {float} -- Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed. (default: {0.2})
        minimum_radius {float} -- Minimum radius of the detected circle given in patch size (default: {0.4})
        maximum_radius {float} -- Maximum radius of the detected citcle given in patch size (default: {0.65})
        max_patch_size {int} -- Scales down the image to this size if it is larger to speed up computation (default: {800})
        max_distance_center {float} -- Maximum distance to the center of the patch given in patch size (default: {0.05})
        param1_limit {int} -- lower bound on param1 during the recursive search if no circle is found (default: {40})
        param1_step {int} -- step size for decreasing param1 in recursive search (default: {10})
        param2_limit {int} -- lower bound on param2 during the recursive search if no circle is found (default: {20})
        param2_step {int} -- step size for decreasing param2 in recursive search (default: {10})

    Returns:
        tuple -- (x, y, r_in, r_out, img) where x,y coordinates of the box center, radius d of the inner box and radius r of the outer box, img image with circles, None if no circle is found
    '''

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    scale_factor = max(gray.shape) / max_patch_size
    shape = (int(gray.shape[0] / scale_factor), int(gray.shape[1] / scale_factor))
    gray = cv2.resize(gray.T, shape).T
    output = None

    minRadius = int(min(gray.shape) * minimum_radius)
    maxRadius = int(min(gray.shape) * maximum_radius)
    minDist = int(min(gray.shape) * minimum_circle_distance)
    maxDist = int(min(gray.shape) * max_distance_center)

    # detect circles in the image
    try:
        circles = cv2.HoughCircles(gray,
                                   cv2.HOUGH_GRADIENT,
                                   dp=dp,
                                   minDist=minDist,
                                   param1=param1,
                                   param2=param2,
                                   minRadius=minRadius,
                                   maxRadius=maxRadius)
    except Exception as e:
        print('Something bad happened:', e)
        return None
    # ensure at least some circles were found

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        center_circle = 0
        if len(circles) > 1:
            cy, cx = np.array(gray.shape) / 2
            dist = np.sqrt((circles[:, 0] - cx)**2 + (circles[:, 1] - cy)**2)
            center_circle = np.argmin(dist)
            if dist[center_circle] > maxDist:
                center_circle = None
                #print(dist[center_circle], maxDist)

        if display:
            # loop over the (x, y) coordinates and radius of the circles
            output = im.copy()
            circles = np.round(scale_factor * circles).astype(int)
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            if center_circle is not None:
                x, y, r = circles[center_circle, :]
                cv2.circle(output, (x, y), r, (0, 255, 255), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (128, 128, 255), -1)

        if center_circle is not None:
            x, y, r_out = np.round(scale_factor * circles[center_circle, :]).astype(int)
            r_in = int(np.sqrt((r_out**2) / 2))
            return x, y, r_in, r_out, output
        else:
            return None

    if circles is None or center_circle is None:
        #warnings.warn('No circles found on image')
        if param1 > param1_limit:
            param1 -= abs(param1_step)
            print('Retry with param1=', param1)
            return find_retina_boxes(im,
                                     display,
                                     dp=dp,
                                     param1=param1,
                                     param2=param2,
                                     minimum_circle_distance=minimum_circle_distance,
                                     minimum_radius=minimum_radius,
                                     maximum_radius=maximum_radius,
                                     max_patch_size=max_patch_size,
                                     max_distance_center=max_distance_center)
        elif param2 > param2_limit:
            param2 -= abs(param2_step)
            print('Retry with param2=', param2)
            return find_retina_boxes(im,
                                     display,
                                     dp=dp,
                                     param1=param1,
                                     param2=param2,
                                     minimum_circle_distance=minimum_circle_distance,
                                     minimum_radius=minimum_radius,
                                     maximum_radius=maximum_radius,
                                     max_patch_size=max_patch_size,
                                     max_distance_center=max_distance_center)
        else:
            print('no luck, skipping image')

    return None


def get_retina_mask(img):
    mask = np.zeros((img.shape[:2]), dtype=np.uint8)
    circle = find_retina_boxes(img)
    if circle is None:
        return mask
    x, y, r_in, r_out, output = circle

    cv2.circle(mask, (x, y), r_out, (255), cv2.FILLED)

    return mask
