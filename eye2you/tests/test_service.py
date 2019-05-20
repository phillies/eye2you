# pylint: disable=redefined-outer-name,protected-access
import os
import pathlib

import torch
import pytest
import numpy as np
from PIL import Image

import eye2you.helper_functions
from eye2you import Coach
from eye2you import SimpleService, CAMService

LOCAL_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture
def checkpoint_filename(tmp_path):
    coach = Coach()
    coach.load_config(LOCAL_DIR / 'data/example.yaml')
    filename = tmp_path / 'test.ckpt'
    coach.save(filename)
    return filename


def test_simpleservice_init(checkpoint_filename):
    service = SimpleService(checkpoint_filename)
    assert service is not None
    assert service.net is not None
    assert service.data_preparation is not None
    assert service.device is not None
    assert isinstance(service.image_size, tuple)


def test_simpleservice_init_error():
    with pytest.raises(ValueError):
        service = SimpleService(None)
        service.initialize()


def test_simpleservice_analyze(checkpoint_filename):
    service = SimpleService(checkpoint_filename)
    img = Image.open(LOCAL_DIR / 'data/classA/img0.jpg')
    res = service.analyze_image(img)
    assert isinstance(res, torch.Tensor)

    img_cv2 = eye2you.helper_functions.PIL_to_cv2(img)
    res2 = service.analyze_image(img_cv2)
    assert isinstance(res, torch.Tensor)
    np.testing.assert_allclose(res.cpu().numpy(), res2.cpu().numpy())

    img_torch = eye2you.helper_functions.PIL_to_torch(img)
    res3 = service.analyze_image(img_torch)
    assert isinstance(res, torch.Tensor)
    np.testing.assert_allclose(res.cpu().numpy(), res3.cpu().numpy())

    with pytest.raises(ValueError):
        _ = service.analyze_image(None)


def test_simpleservice_classifyall(checkpoint_filename):
    service = SimpleService(checkpoint_filename)
    files = [LOCAL_DIR / 'data/classA/img0.jpg', LOCAL_DIR / 'data/classB/img2.jpg']
    res = service.classify_all(files)
    assert isinstance(res, torch.Tensor)
    assert res.shape[0] == len(files)
    for ii in range(len(files)):
        np.testing.assert_allclose(res[ii, ...], service.analyze_image(Image.open(files[ii])))


def test_simpleservice_print(checkpoint_filename):
    service = SimpleService(checkpoint_filename)
    target = """eye2you Service:
Loaded from {}
Network: inception_v3_xs
Transform:
Compose(
    Resize(size=200, interpolation=PIL.Image.BILINEAR)
    CenterCrop(size=(200, 200))
    ToTensor()
    Normalize(mean=[0.3198, 0.1746, 0.0901], std=[0.2287, 0.1286, 0.0723])
)
""".format(checkpoint_filename)
    assert str(service) == target


def test_camservice_init(checkpoint_filename):
    service = CAMService(checkpoint_filename)
    assert service.num_classes > 0
    assert service._finalconv_name is not None
    assert service._weight_softmax is not None


def test_camservice_cam(checkpoint_filename):
    service = CAMService(checkpoint_filename)
    img = Image.open(LOCAL_DIR / 'data/classA/img0.jpg')

    cams = service.get_class_activation_map(img)
    assert isinstance(cams, list)
    for cam in cams:
        assert isinstance(cam, Image.Image)
        assert cam.size == img.size

    cams = service.get_class_activation_map(img, 0)
    assert len(cams) == 1
    assert isinstance(cams, list)
    for cam in cams:
        assert isinstance(cam, Image.Image)
        assert cam.size == img.size

    cams = service.get_class_activation_map(img, (0, 1))
    assert len(cams) == 2
    assert isinstance(cams, list)
    for cam in cams:
        assert isinstance(cam, Image.Image)
        assert cam.size == img.size

    cams = service.get_class_activation_map(img, 0, as_pil_image=False)
    assert len(cams) == 1
    assert isinstance(cams, list)
    for cam in cams:
        assert isinstance(cam, np.ndarray)
        assert cam.shape == img.size

    with pytest.raises(ValueError):
        _ = service.get_class_activation_map(img, '0')
