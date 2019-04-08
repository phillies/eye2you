import os
import sys
import numpy as np
from PIL import Image
import torch

# Functions partially copied from torchvision

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset(directory, class_to_idx, extensions):
    images = []
    directory = os.path.expanduser(directory)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(directory, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.relpath(os.path.join(root, fname), directory)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def get_images(directory, extensions):
    images = sorted([
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if has_file_allowed_extension(os.path.join(directory, d))
    ])
    return images


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(directory):
    """
    Finds the class folders in a dataset.
    Args:
        directory (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (directiry), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    print(sys.version_info)
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def merge_models_from_checkpoints(checkpoints, device='cpu'):
    '''Extracts the model state_dict from a list of RetinaChecker checkpoints
    and returns the dictionaly with all models, example config and classes.
    Returns the first config and classes it finds. Make sure the checkpoints are
    compatible. Otherwise somewhere an exception will rise.

    Arguments:
        checkpoints {tuple} -- Filenames of RetinaChecker checkpoints (or torch checkpoints)

    Returns
        dict -- Dictionary with models, config, and classes
    '''
    models = []
    config = None
    classes = None
    for filename in checkpoints:
        data = torch.load(filename, map_location=device)
        if 'state_dict' in data.keys():
            models.append(data['state_dict'])
        if config is None and 'config' in data.keys():
            config = data['config']
        if classes is None and 'classes' in data.keys():
            classes = data['classes']

    return {'models': models, 'config': config, 'classes': classes}