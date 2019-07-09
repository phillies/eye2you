import sys
import cv2
import numpy as np
import torch
from PIL import Image

from .net import Network
from .helper_functions import cv2_to_PIL, torch_to_PIL
from . import datasets

if 'IPython' in sys.modules:

    from IPython import get_ipython

    if 'IPKernelApp' in get_ipython().config:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
else:
    from tqdm import tqdm

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


# def majority_vote(predictions):
#     result = (np.count_nonzero(predictions, axis=2) > predictions.shape[2] / 2) * 1.0
#     return result


class BaseService():

    def initialize(self):
        raise NotImplementedError()

    def analyze_image(self, img):
        raise NotImplementedError()


class SimpleService(BaseService):

    def __init__(self, checkpoint, device=None):
        self.net = None
        self.checkpoint = checkpoint
        self.last_result = None

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

    def analyze_image(self, img):
        if isinstance(img, np.ndarray):
            image = cv2_to_PIL(img)
        elif isinstance(img, torch.Tensor):
            image = torch_to_PIL(img)
        elif isinstance(img, Image.Image):
            image = img
        else:
            raise ValueError('Only PIL Image or numpy array supported')

        # Convert image to tensor
        x_input = self.data_preparation.transform(image)

        #Reshape for input intp 1,n,h,w
        x_input = x_input.unsqueeze(0)

        with torch.no_grad():
            output = self.net.model(x_input.to(self.net.device)).cpu()

        self.last_result = output.squeeze()
        return output.squeeze()

    def classify_all(self, filenames):
        outputs = []
        for fname in tqdm(filenames, desc='Files', position=0):
            img = Image.open(fname)
            outputs.append(self.analyze_image(img))
        return torch.stack(outputs, dim=0)

    def __str__(self):
        desc = 'eye2you Service:\n'
        desc += 'Loaded from {}\n'.format(self.checkpoint)
        desc += 'Network: ' + str(self.net.model_name)
        desc += '\nTransform:\n' + str(self.data_preparation.get_transform()) + '\n'
        return desc


class CAMService(SimpleService):

    def __init__(self, checkpoint, device=None):
        self._feature_extractor_hook = None
        self.num_classes = 0
        self._final_conv_name = None
        self._weight_softmax = None
        super().__init__(checkpoint, device)

    def initialize(self):
        super().initialize()

        self.num_classes = list(self.net.model.children())[-1].out_features
        # This is the initialization of the class activation map extraction
        self._finalconv_name = list(self.net.model.named_children())[-2][0]
        params = list(self.net.model.parameters())
        self._weight_softmax = np.squeeze(params[-2].data.detach().cpu().numpy())

    def hook_feature_extractor(self):
        # hook the feature extractor
        self._feature_extractor_hook = self.net.model._modules.get(self._finalconv_name).register_forward_hook(  # pylint: disable=protected-access
            hook_feature)

    def unhook_feature_extractor(self):
        if self._feature_extractor_hook is not None:
            self._feature_extractor_hook.remove()

    # def get_largest_prediction(self, image):
    #     '''Returns the class index of the largest prediction

    #     Arguments:
    #         image {PIL.Image.Image} -- PIL image to analyze

    #     Returns:
    #         int -- class index of the largest prediction
    #     '''
    #     pred = self.analyze_image(image)
    #     return int(pred.argmax())

    def get_class_activation_map(self,
                                 image,
                                 single_cam=None,
                                 as_pil_image=True,
                                 min_threshold=None,
                                 max_threshold=None):
        # get the softmax weight
        # params = list(self.net.model.parameters())
        # weight_softmax = np.squeeze(params[-2].data.detach().cpu().numpy())

        # calculating the FEATURE_BLOBS
        self.hook_feature_extractor()
        self.analyze_image(image)
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

        CAMs = returnCAM(FEATURE_BLOBS[-1], self._weight_softmax, idx, image.size)
        if as_pil_image:
            for ii, cam in enumerate(CAMs):
                CAMs[ii] = cv2_to_PIL(cam, min_threshold, max_threshold)
        return CAMs

    # def get_contour(self, img, threshold=10, camId=None, crop_black_borders=True, border_threshold=20):
    #     if isinstance(img, np.ndarray):
    #         image = cv2_to_PIL(img)
    #     if isinstance(img, torch.Tensor):
    #         image = torch_to_PIL(img)
    #     elif isinstance(img, Image.Image):
    #         image = img
    #     else:
    #         raise ValueError('Only PIL Image or numpy array supported')

    #     CAMs = self.get_class_activation_map(image, single_cam=camId, as_pil_image=False)
    #     if camId is None:
    #         camId = int(self.last_result.argmax())
    #         CAMs = [CAMs[camId]]

    #     # scaling CAM to input image size
    #     cam_mask = cv2.resize(CAMs[0], dsize=image.size, interpolation=cv2.INTER_CUBIC)

    #     # cropping away the black borders
    #     if crop_black_borders:
    #         img_mask = (np.array(image).max(2) > border_threshold)
    #         cam_mask = cam_mask * img_mask

    #     # thresholding
    #     cam_mask = ((cam_mask > threshold) * 255).astype(np.ubyte)

    #     # Contour detection
    #     contours_return = cv2.findContours(cam_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # opencv changed return value from (image, contours, hierarchy) in 3.4 to (contours, hierarchy) in 4.0
    #     if len(contours_return) == 3:
    #         _, contours, _ = contours_return
    #     elif len(contours_return) == 2:
    #         contours, _ = contours_return
    #     else:
    #         message = '''cv2.findCountours() returned {} values. This version can only deal with 2 or 3 return values (tested on cv2 3.4 and 4.0).
    #         Your version: {}
    #         Please submit a bug report with your opencv version and/or switch to 3.4 or 4.x in the meantime.'''.format(
    #             len(contours_return), cv2.__version__)
    #         raise RuntimeError(message)

    #     return contours
