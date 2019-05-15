# pylint: disable=redefined-outer-name
import os
import pathlib

import pytest
import configparser
import numpy as np
from PIL import Image

import eye2you.helper_functions
from eye2you import Service

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
