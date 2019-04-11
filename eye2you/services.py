import sys
import configparser
import functools

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from .checker import RetinaChecker
from .datasets import PandasDataset
from .io_helper import merge_models_from_checkpoints
from .image_helper import cv2_to_PIL, PIL_to_cv2, torch_to_cv2, float_to_uint8, find_retina_boxes
from .models import u_net

FEATURE_BLOBS = []

def denormalize_transform(trans):
    for t in trans.transforms:
        if isinstance(t, torchvision.transforms.transforms.Normalize):
            return denormalize(t)
    return None

def denormalize(normalize):
    std = 1 / np.array(normalize.std)
    mean = np.array(normalize.mean) * -1 * std
    denorm = torchvision.transforms.Normalize(mean=mean, std=std)
    return denorm

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


class Service():

    def __init__(self, checkpoint=None, device=None):
        self.retina_checker = None
        self.checkpoint = checkpoint
        self.transform = None
        self.model_image_size = None
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
        self.retina_checker = RetinaChecker(self.device)
        self.retina_checker.initialize(self.checkpoint)
        self.retina_checker.initialize_model()
        self.retina_checker.initialize_criterion()
        self.retina_checker.load_state(self.checkpoint)

        self.retina_checker.model.eval()

        self.model_image_size = self.retina_checker.image_size
        self.num_classes = self.retina_checker.num_classes

        # This is the factor that the image will be scaled to before cropping the center
        # for the model. Empirically a factor between 1.0 and 1.1 yielded the best results
        # as if further reduces possible small boundaries and focuses on the center of the image
        self.test_image_size_overscaling = 1.0

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(self.model_image_size * self.test_image_size_overscaling)),
            torchvision.transforms.CenterCrop(self.model_image_size),
            torchvision.transforms.ToTensor()
        ])
        if self.retina_checker.normalize_factors is not None:
            self.transform = torchvision.transforms.Compose([
                self.transform,
                torchvision.transforms.Normalize(self.retina_checker.normalize_mean, self.retina_checker.normalize_std)
            ])

        # This is the initialization of the class activation map extraction
        # Yes, accessing protected members is not a good style. We'll fix that later ;-)
        self.finalconv_name = list(self.retina_checker.model._modules.keys())[-2]  #pylint: disable=protected-access

    def hook_feature_extractor(self):
        # hook the feature extractor
        self.feature_extractor_hook = self.retina_checker.model._modules.get(self.finalconv_name).register_forward_hook(
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
        x_input = self.transform(image)

        #Reshape for input intp 1,n,h,w
        x_input = x_input.unsqueeze(0)

        return self._classify(x_input).squeeze()

    def validate(self, file_list, root='', num_workers=0, batch_size=None):
        dataset = PandasDataset(source=file_list, mode='csv', root=root, transform=self.transform)
        if batch_size is None:
            batch_size = self.retina_checker.config['hyperparameter'].getint('batch size', 32)

        test_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=num_workers)

        self.last_dataset = dataset
        self.last_loader = test_loader

        return self.retina_checker.validate(test_loader=test_loader)

    def classify_images(self, file_list, root='', output_return=None, num_workers=0, batch_size=None):
        dataset = PandasDataset(source=file_list, mode='csv', root=root, transform=self.transform)
        if batch_size is None:
            batch_size = self.retina_checker.config['hyperparameter'].getint('batch size', 32)

        test_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=num_workers)

        result = np.empty((len(dataset), self.num_classes))
        output_buffer = np.empty((len(dataset), self.num_classes))
        self.retina_checker.model.eval()
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            counter = 0
            for images, labels in test_loader:
                images = images.to(self.retina_checker.device)
                #labels = labels.to(self.retina_checker.device)

                outputs = self.retina_checker.model(images)
                #loss = self.retina_checker.criterion(outputs, labels)

                predicted = torch.nn.Sigmoid()(outputs).round()
                num_images = predicted.size()[0]
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
            output = self.retina_checker.model(x_input.to(self.retina_checker.device))
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
        params = list(self.retina_checker.model.parameters())
        weight_softmax = np.squeeze(params[-2].data.detach().cpu().numpy())

        # calculating the FEATURE_BLOBS
        self.hook_feature_extractor()
        self.classify_image(image)
        self.unhook_feature_extractor()

        if single_cam is None:
            idx = np.arange(self.num_classes, dtype=np.int)
        elif isinstance(single_cam, int):
            idx = [single_cam]
        elif isinstance(single_cam, tuple) or isinstance(single_cam, list):
            idx = single_cam
        else:
            raise ValueError('single_cam not recognized as None, int, or tuple: {} with type {}'.format(
                single_cam, type(single_cam)))

        CAMs = returnCAM(FEATURE_BLOBS[-1], weight_softmax, idx, (self.model_image_size, self.model_image_size))
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
        desc += 'RetinaChecker:\n' + self.retina_checker._str_core_info()  # pylint disable:protected-access
        desc += 'Transform:\n' + str(self.transform)
        return desc

    def print_module_versions():
        versions = ''
        versions += 'Python: ' + sys.version
        versions += '\nNumpy: ' + np.__version__
        versions += '\nPIL: ' + Image.__version__
        versions += '\nTorch: ' + torch.__version__
        versions += '\nTorchvision: ' + torchvision.__version__
        versions += '\nOpenCV: ' + cv2.__version__
        print(versions)


class MEService(Service):

    def __init__(self, mixture_checkpoint=None, device=None):
        super().__init__(checkpoint=None)
        self.checkpoint = mixture_checkpoint
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if mixture_checkpoint is not None:
            self.initialize()

    def initialize(self):
        if self.checkpoint is None:
            raise ValueError('checkpoint attribute must be set')

        if isinstance(self.checkpoint, (list, tuple)):
            data = merge_models_from_checkpoints(self.checkpoint, self.device)
        else:
            data = torch.load(self.checkpoint, map_location=self.device)
        if 'models' not in data.keys() or 'config' not in data.keys() or 'classes' not in data.keys():
            raise ValueError('Checkpoint must contain models, config, and classes keys')

        self.number_of_experts = len(data['models'])
        self.retina_checker = []
        self.config = configparser.ConfigParser()
        self.config.read_string(data['config'])

        for ii in range(self.number_of_experts):
            rc = RetinaChecker(self.device)
            rc.initialize(self.config)
            rc.classes = data['classes']
            rc.initialize_model()
            rc.initialize_criterion()
            rc.model.load_state_dict(data['models'][ii], strict=False)
            rc.model.eval()

            # This is the initialization of the class activation map extraction
            finalconv_name = list(rc.model._modules.keys())[-2]

            self.retina_checker.append(rc)

        self.model_image_size = self.retina_checker[0].image_size
        self.num_classes = len(data['classes'])
        self.mixture_function = majority_vote

        # This is the factor that the image will be scaled to before cropping the center
        # for the model. Empirically a factor between 1.0 and 1.1 yielded the best results
        # as if further reduces possible small boundaries and focuses on the center of the image
        self.test_image_size_overscaling = 1.0

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(self.model_image_size * self.test_image_size_overscaling)),
            torchvision.transforms.CenterCrop(self.model_image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.retina_checker[0].normalize_mean,
                                             self.retina_checker[0].normalize_std)
        ])

    def _classify(self, x_input, output_return=None):
        with torch.no_grad():
            pred = []
            outputs = []
            for ii in range(self.number_of_experts):
                output = self.retina_checker[ii].model(x_input.to(self.retina_checker[ii].device))
                outputs.append(output)
                pred.append(torch.nn.Sigmoid()(output).detach().cpu().numpy())
            prediction = np.array(pred)
            if isinstance(output_return, list):
                output_return.append(outputs)
        return prediction

    def get_largest_prediction(self, image):
        '''Returns the class index of the largest prediction

        Arguments:
            image {PIL.Image.Image} -- PIL image to analyze

        Returns:
            int -- class index of the largest prediction
        '''
        pred = self.classify_image(image)
        max_preds = pred.argmax(axis=1)
        count, _ = np.histogram(max_preds, np.arange(0, pred.shape[1] + 1))
        return int(count.argmax())

    def classify_images(self, file_list, root='', output_return=None, num_workers=0, batch_size=None):
        dataset = PandasDataset(source=file_list, mode='csv', root=root, transform=self.transform)
        if batch_size is None:
            batch_size = self.retina_checker[0].config['hyperparameter'].getint('batch size', 32)

        test_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, sampler=None, num_workers=num_workers)

        result = np.empty((len(dataset), self.num_classes))
        output_buffer = np.empty((len(dataset), self.num_classes, self.number_of_experts))
        for rc in self.retina_checker:
            rc.model.eval()
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            counter = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                #labels = labels.to(self.retina_checker.device)

                num_images = len(images)
                all_pred = np.empty((num_images, self.num_classes, self.number_of_experts))
                for (ii, rc) in enumerate(self.retina_checker):
                    outputs = rc.model(images)
                    output_buffer[counter:counter + num_images, :, ii] = outputs.cpu().numpy()
                    all_pred[:, :, ii] = torch.nn.Sigmoid()(outputs).round().cpu().numpy()

                predicted = self.mixture_function(all_pred)
                result[counter:counter + num_images, :] = predicted
                counter += num_images

        self.last_dataset = dataset
        self.last_loader = test_loader
        if isinstance(output_return, list):
            output_return.append(output_buffer)
        return result


class MultiService(Service):

    def __init__(self, checkpoint=None, device=None, unet_depth=2, unet_state=None):
        super().__init__(checkpoint=None, device=device)
        self.checkpoint = checkpoint
        self.unet = u_net(in_channels=3, out_channels=2, depth=unet_depth).to(self.device)
        self.unet_depth = unet_depth
        self.unet_state = unet_state
        if checkpoint is not None and unet_state is not None:
            self.initialize()

    def initialize(self):
        if self.unet_state is None:
            raise ValueError('unet_state cannot be None')
        super().initialize()

        self.unet.load_state_dict(torch.load(self.unet_state, map_location=self.device), strict=False)

    def get_vessels(self, img, merge_image=False, img_size=320, **kwargs):
        #apply padding so that it is divisible by 2 if size < 512, else use sliding windows of 256?!?
        if isinstance(img, np.ndarray):
            image = cv2_to_PIL(img)
        elif isinstance(img, Image.Image):
            image = img
        else:
            raise ValueError('Only PIL Image or numpy array supported')

        print('DEMO MODE: Rescaling image to 320x320 before vessel detection')
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(img_size)),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.retina_checker.normalize_mean, self.retina_checker.normalize_std)
        ])
        denorm = denormalize_transform(transform)
        x_img = transform(image)

        #x_img = self.transform(image)
        x_device = x_img.to(self.device).unsqueeze(0)

        y_device = self.unet(x_device)
        y_img = y_device.to('cpu')
        del x_device, y_device
        label = (y_img[:, 1, :, :] > y_img[:, 0, :, :]).float().squeeze()

        #mask = self.get_retina_mask(float_to_uint8(torch_to_cv2(denorm(x_img))), **kwargs)
        mask = self.get_retina_mask(PIL_to_cv2(image.resize((label.shape[1], label.shape[0]))), **kwargs)
        
        vessel = cv2_to_PIL(label.numpy() * mask)
        vessel = torchvision.transforms.Resize(image.size, interpolation=Image.NEAREST)(vessel)

        if merge_image:
            img_cv = PIL_to_cv2(img).astype(float)
            ves_cv = PIL_to_cv2(vessel).astype(float)
            out = np.clip(img_cv + ves_cv[:, :, None], 0, 255)
            vessel = cv2_to_PIL(out)
        return vessel

    def get_retina_mask(self, img, **kwargs):
        mask = np.zeros((img.shape[:2]), dtype=np.uint8)
        circle = find_retina_boxes(
            img,
            display=False,
            **kwargs)
        if circle is None:
            mask = mask + 1
        else:
            x, y, r_in, r_out, output = circle

            cv2.circle(mask, (x, y), r_out - 1, (255), cv2.FILLED)
        return mask