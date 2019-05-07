import sys
import configparser
import functools

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from .net import Network
from .helper_functions import cv2_to_PIL, PIL_to_cv2, torch_to_cv2, float_to_uint8, find_retina_boxes, denormalize_mean_std, get_retina_mask
from . import factory
from . import datasets

FEATURE_BLOBS = []


def hook_feature(_module, _input, out, ii=0):
    FEATURE_BLOBS.append(out.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample=(256, 256), inter=cv2.INTER_LINEAR):
    # generate the class activation maps upsample to 256x256

    _, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        #cam = cam - np.min(cam)
        #cam_img = cam / np.max(cam)
        #cam_img = np.uint8(255 * cam_img)
        #cam_img = Image.fromarray(cam_img)
        if size_upsample is not None:
            #cam_img = cam_img.resize(size_upsample, resample=Image.BILINEAR)
            cam = cv2.resize(cam, size_upsample, interpolation=inter)
        output_cam.append(cam)
    return output_cam


def majority_vote(predictions):
    result = (np.count_nonzero(predictions, axis=2) > predictions.shape[2] / 2) * 1.0
    return result

class SimpleService():

    def __init__(self, checkpoint, device=None):
        self.net = None
        self.checkpoint = checkpoint

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if checkpoint is not None:
            self.initialize()

    def initialize(self):
        if self.checkpoint is None:
            raise ValueError('checkpoint cannot be None')
        ckpt = torch.load(self.checkpoint, map_location=self.device)
        self.net = Network.from_state_dict(ckpt, self.device)

        self.net.model.eval()

        self.data_preparation = datasets.DataPreparation(**ckpt['config']['data_preparation'])

        # image size is in PIL format (width, height)!
        if isinstance(self.data_preparation.size, (tuple, list)):
            self.image_size = (self.data_preparation.size[1], self.data_preparation.size[0])
        else:
            self.image_size = (self.data_preparation.size, self.data_preparation.size)

    def classify_image(self, img):
        if isinstance(img, np.ndarray):
            image = cv2_to_PIL(img)
        elif isinstance(img, Image.Image):
            image = img
        else:
            raise ValueError('Only PIL Image or numpy array supported')

        # Convert image to tensor
        x_input = self.data_preparation.get_transform()(image)

        #Reshape for input intp 1,n,h,w
        x_input = x_input.unsqueeze(0)

        with torch.no_grad():
            output = self.net.model(x_input.to(self.net.device))

        return output.squeeze()



class Service():

    def __init__(self, checkpoint=None, device=None):
        self.net = None
        self.vesselnet = None
        self.checkpoint = checkpoint

        self.test_image_size_overscaling = None

        self.last_dataset = None
        self.last_loader = None

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.feature_extractor_hook = None

        if checkpoint is not None:
            self.initialize()

    def initialize(self):
        if self.checkpoint is None:
            raise ValueError('checkpoint cannot be None')
        ckpt = torch.load(self.checkpoint, map_location=self.device)
        self.net = Network.from_state_dict(ckpt, self.device)
        if 'vessel' in ckpt:
            self.vesselnet = Network.from_state_dict(ckpt['vessel'], self.device)
            self.vesselnet.model.eval()

        self.net.model.eval()

        self.num_classes = list(self.net.model.children())[-1].out_features

        # This is the factor that the image will be scaled to before cropping the center
        # for the model. Empirically a factor between 1.0 and 1.1 yielded the best results
        # as if further reduces possible small boundaries and focuses on the center of the image
        self.test_image_size_overscaling = 1.0

        self.data_preparation = datasets.DataPreparation(**ckpt['config']['data_preparation'])

        # image size is in PIL format (width, height)!
        if isinstance(self.data_preparation.size, (tuple, list)):
            self.image_size = (self.data_preparation.size[1], self.data_preparation.size[0])
        else:
            self.image_size = (self.data_preparation.size, self.data_preparation.size)

        # This is the initialization of the class activation map extraction
        self.finalconv_name = list(self.net.model.named_children())[-2][0]

    def hook_feature_extractor(self):
        # hook the feature extractor
        self.feature_extractor_hook = self.net.model._modules.get(self.finalconv_name).register_forward_hook(
            hook_feature)  #pylint: disable=protected-access

    def unhook_feature_extractor(self):
        if self.feature_extractor_hook is not None:
            self.feature_extractor_hook.remove()

    def classify_image(self, img):
        if isinstance(img, np.ndarray):
            image = cv2_to_PIL(img)
        elif isinstance(img, Image.Image):
            image = img
        else:
            raise ValueError('Only PIL Image or numpy array supported')

        # Convert image to tensor
        x_input = self.data_preparation.get_transform()(image)

        #Reshape for input intp 1,n,h,w
        x_input = x_input.unsqueeze(0)

        return self._classify(x_input).squeeze()

    def classify_images(self, file_list, root='', output_return=None, num_workers=0, batch_size=1):
        dataset = datasets.TripleDataset(*factory.load_csv(file_list, root), preparation=self.data_preparation)

        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  sampler=None,
                                                  num_workers=num_workers)

        result = np.empty((len(dataset), self.num_classes))
        output_buffer = np.empty((len(dataset), self.num_classes))
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            counter = 0
            for images, _ in test_loader:
                images = images.to(self.net.device)
                #labels = labels.to(self.retina_checker.device)

                outputs = self.net.model(images)
                #loss = self.retina_checker.criterion(outputs, labels)

                predicted = torch.nn.Sigmoid()(outputs).round()
                num_images = predicted.shape[0]
                result[counter:counter + num_images, :] = predicted.cpu().numpy()
                output_buffer[counter:counter + num_images, :] = outputs.cpu().numpy()
                counter += num_images

        self.last_dataset = dataset
        self.last_loader = test_loader
        if isinstance(output_return, list):
            output_return.append(output_buffer)
        return result

    def _classify(self, x_input, output_return=None):
        with torch.no_grad():
            output = self.net.model(x_input.to(self.net.device))
            prediction = torch.nn.Sigmoid()(output).detach().cpu().numpy()
            if isinstance(output_return, list):
                output_return.append(output)
        return prediction

    def get_largest_prediction(self, image):
        '''Returns the class index of the largest prediction

        Arguments:
            image {PIL.Image.Image} -- PIL image to analyze

        Returns:
            int -- class index of the largest prediction
        '''
        pred = self.classify_image(image)
        return int(pred.argmax())

    def get_class_activation_map(self,
                                 image,
                                 single_cam=None,
                                 as_pil_image=True,
                                 min_threshold=None,
                                 max_threshold=None):
        # get the softmax weight
        params = list(self.net.model.parameters())
        weight_softmax = np.squeeze(params[-2].data.detach().cpu().numpy())

        # calculating the FEATURE_BLOBS
        self.hook_feature_extractor()
        self.classify_image(image)
        self.unhook_feature_extractor()

        if single_cam is None:
            idx = np.arange(self.num_classes, dtype=np.int)
        elif isinstance(single_cam, int):
            idx = [single_cam]
        elif isinstance(single_cam, (tuple, list)):
            idx = single_cam
        else:
            raise ValueError('single_cam not recognized as None, int, or tuple: {} with type {}'.format(
                single_cam, type(single_cam)))

        CAMs = returnCAM(FEATURE_BLOBS[-1], weight_softmax, idx, self.image_size)
        if as_pil_image:
            for ii, cam in enumerate(CAMs):
                CAMs[ii] = cv2_to_PIL(cam, min_threshold, max_threshold)
        return CAMs

    def get_contour(self, img, threshold=10, camId=None, crop_black_borders=True, border_threshold=20):
        if isinstance(img, np.ndarray):
            image = cv2_to_PIL(img)
        elif isinstance(img, Image.Image):
            image = img
        else:
            raise ValueError('Only PIL Image or numpy array supported')

        if camId is None:
            camId = self.get_largest_prediction(image)
        CAMs = self.get_class_activation_map(image, single_cam=camId, as_pil_image=False)

        # scaling CAM to input image size
        cam_mask = cv2.resize(CAMs[0], dsize=image.size, interpolation=cv2.INTER_CUBIC)

        # cropping away the black borders
        if crop_black_borders:
            img_mask = (np.array(image).max(2) > border_threshold)
            cam_mask = cam_mask * img_mask

        # thresholding
        cam_mask = ((cam_mask > threshold) * 255).astype(np.ubyte)

        # Contour detection
        contours_return = cv2.findContours(cam_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # opencv changed return value from (image, contours, hierarchy) in 3.4 to (contours, hierarchy) in 4.0
        if len(contours_return) == 3:
            _, contours, _ = contours_return
        elif len(contours_return) == 2:
            contours, _ = contours_return
        else:
            message = '''cv2.findCountours() returned {} values. This version can only deal with 2 or 3 return values (tested on cv2 3.4 and 4.0).
            Your version: {}
            Please submit a bug report with your opencv version and/or switch to 3.4 or 4.0 in the meantime.'''.format(
                len(contours_return), cv2.__version__)
            raise RuntimeError(message)

        return contours

    def __str__(self):
        desc = 'medAI Service:\n'
        desc += 'Loaded from {}\n'.format(self.checkpoint)
        desc += 'Network:\n' + str(self.net.model_name)
        desc += 'Transform:\n' + str(self.data_preparation.get_transform())
        return desc

    @staticmethod
    def print_module_versions():
        versions = ''
        versions += 'Python: ' + sys.version
        versions += '\nNumpy: ' + np.__version__
        versions += '\nPIL: ' + Image.__version__
        versions += '\nTorch: ' + torch.__version__
        versions += '\nTorchvision: ' + torchvision.__version__
        versions += '\nOpenCV: ' + cv2.__version__
        print(versions)

    def get_vessels(self, img, merge_image=False, **kwargs):
        #apply padding so that it is divisible by 2 if size < 512, else use sliding windows of 256?!?
        if isinstance(img, np.ndarray):
            image = cv2_to_PIL(img)
        elif isinstance(img, Image.Image):
            image = img
        else:
            raise ValueError('Only PIL Image or numpy array supported')

        print('DEMO MODE: Rescaling image to 320x320 before vessel detection')
        transform = self.data_preparation.get_transform()

        x_img = transform(image)

        #x_img = self.transform(image)
        x_device = x_img.to(self.device).unsqueeze(0)

        with torch.no_grad():
            y_device = self.vesselnet.model(x_device)
        y_img = y_device.to('cpu')
        del x_device, y_device
        label = y_img.round().squeeze()

        #mask = self.get_retina_mask(float_to_uint8(torch_to_cv2(denorm(x_img))), **kwargs)
        mask = get_retina_mask(PIL_to_cv2(image.resize((label.shape[1], label.shape[0]))), **kwargs)

        vessel = cv2_to_PIL(label.numpy() * mask)
        vessel = torchvision.transforms.Resize(image.size, interpolation=Image.NEAREST)(vessel)

        if merge_image:
            img_cv = PIL_to_cv2(img).astype(float)
            ves_cv = PIL_to_cv2(vessel).astype(float)
            out = np.clip(img_cv + ves_cv[:, :, None], 0, 255)
            vessel = cv2_to_PIL(out)
        return vessel
