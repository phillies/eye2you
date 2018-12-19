import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from .RetinaCheckerPandas import RetinaCheckerPandas as RetinaChecker

def PIL_to_cv2(img):
    return np.array(img)

def cv2_to_PIL(img, min_val=None, max_val=None):
    if min_val is None:
        min_val = img.min()
    if max_val is None:
        max_val = img.max()
    img_scale = img - min_val
    img_scale = img_scale / (max_val - min_val)
    img_scale = (img_scale * 255).astype(np.ubyte)
    pil_img = Image.fromarray(img_scale)
    return pil_img


features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample = (256, 256), inter=cv2.INTER_LINEAR):
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

class Service():

    def __init__(self, checkpoint):
        self.retina_checker = None
        self.checkpoint = checkpoint
        self.transform = None
        self.model_image_size = None
        self.test_image_size_overscaling = None

        self.retina_checker = RetinaChecker()
        self.retina_checker.initialize(self.checkpoint)
        self.retina_checker.initialize_model()
        self.retina_checker.initialize_criterion()
        self.retina_checker.load_state(self.checkpoint)

        self.retina_checker.model.eval()

        self.model_image_size = self.retina_checker.image_size

        # This is the factor that the image will be scaled to before cropping the center
        # for the model. Empirically a factor between 1.0 and 1.1 yielded the best results
        # as if further reduces possible small boundaries and focuses on the center of the image
        self.test_image_size_overscaling = 1.0

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(self.model_image_size * self.test_image_size_overscaling)),
            torchvision.transforms.CenterCrop(self.model_image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.retina_checker.normalize_mean, self.retina_checker.normalize_std)
        ])

        # This is the initialization of the class activation map extraction
        
        finalconv_name = list(self.retina_checker.model._modules.keys())[-2]
        # hook the feature extractor

        self.retina_checker.model._modules.get(finalconv_name).register_forward_hook(hook_feature)
    
    def classify_image(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError('Only PIL images supported by now')

        # Convert image to tensor
        x_input = self.transform(image)

        #Reshape for input intp 1,n,h,w
        x_input = x_input.unsqueeze(0)

        return self._classify(x_input).squeeze()

    def _classify(self, x_input):
        with torch.no_grad():
            output = self.retina_checker.model(x_input.to(self.retina_checker.device))
            prediction = torch.nn.Sigmoid()(output).detach().cpu().numpy()
        
        return prediction
    
    def get_largest_prediction(self, image):
        pred = self.classify_image(image)
        return pred.argmax()

    
    def get_class_activation_map(self, image, single_cam=None, as_pil_image=True, min_threshold=None, max_threshold=None):
        # get the softmax weight
        params = list(self.retina_checker.model.parameters())
        weight_softmax = np.squeeze(params[-2].data.detach().cpu().numpy())
        
        # calculating the features_blobs
        self.classify_image(image)

        if single_cam is None:
            idx = np.arange(self.retina_checker.num_classes, dtype=np.int)
        elif isinstance(single_cam, int):
            idx = [single_cam]
        elif isinstance(single_cam, tuple) or isinstance(single_cam, list):
            idx = single_cam
        else:
            raise ValueError('single_cam not recognized as None, int, or tuple')
        
        CAMs = returnCAM(features_blobs[-1], weight_softmax, idx, (self.model_image_size, self.model_image_size))
        if as_pil_image:
            for ii, cam in enumerate(CAMs):
                CAMs[ii] = cv2_to_PIL(cam, min_threshold, max_threshold)
        return CAMs

    def get_contour(self, image, threshold=10, camId=None, crop_black_borders=True, border_threshold=20):
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
        _, contours, _ = cv2.findContours(cam_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours
        
    
    def __str__(self):
        desc = 'medAI Service:\n'
        desc += 'Loaded from {}\n'.format(self.checkpoint)
        desc += 'RetinaChecker:\n' + self.retina_checker._str_core_info()
        desc += 'Transform:\n' + str(self.transform)
        return desc
