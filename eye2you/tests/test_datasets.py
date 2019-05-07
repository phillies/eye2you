# pylint: disable=redefined-outer-name
import os
import pathlib

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from eye2you.datasets import TripleDataset, DataAugmentation, DataPreparation
from PIL import Image

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
NUMBER_OF_CLASSES = 2
NUMBER_OF_IMAGES = 4
