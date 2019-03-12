from .RetinaChecker import RetinaChecker
from .Service import Service
import torch
import torchvision
import configparser
import functools
from PIL import Image
import numpy as np

features_blobs = []
def hook_feature(_module, _input, out, ii):
    features_blobs[ii].append(out.data.cpu().numpy())

class MEService(Service):

    def __init__(self, mixture_checkpoint=None, device=None):
        super().__init__(checkpoint=None)
        self.checkpoint = mixture_checkpoint
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        if mixture_checkpoint is not None:
            self.initialize()
        
    def initialize(self, device=None):
        if self.checkpoint is None:
            raise ValueError('checkpoint attribute must be set')

        data = torch.load(self.checkpoint, map_location=self.device)
        if 'models' not in data.keys() or 'config' not in data.keys() or 'classes' not in data.keys():
            raise ValueError('Checkpoint must contain models, config, and classes keys')
            
        self.number_of_experts = len(data['models'])
        self.retina_checker = []
        self.config = configparser.ConfigParser()
        self.config.read_string(data['config'])

        for ii in range(self.number_of_experts):
            rc = RetinaChecker()
            rc.initialize(self.config)
            rc.classes = data['classes']
            rc.initialize_model()
            rc.initialize_criterion()
            rc.model.load_state_dict(data['models'][ii], strict=False)
            rc.model.eval()

            # This is the initialization of the class activation map extraction
            finalconv_name = list(rc.model._modules.keys())[-2]
            # hook the feature extractor
            rc.model._modules.get(finalconv_name).register_forward_hook(functools.partial(hook_feature, ii=ii))

            self.retina_checker.append(rc)

        self.model_image_size = self.retina_checker[0].image_size

        # This is the factor that the image will be scaled to before cropping the center
        # for the model. Empirically a factor between 1.0 and 1.1 yielded the best results
        # as if further reduces possible small boundaries and focuses on the center of the image
        self.test_image_size_overscaling = 1.0

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(self.model_image_size * self.test_image_size_overscaling)),
            torchvision.transforms.CenterCrop(self.model_image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.retina_checker[0].normalize_mean, self.retina_checker[0].normalize_std)
        ])

    def _classify(self, x_input):
        with torch.no_grad():
            pred = []
            for ii in range(self.number_of_experts):
                output = self.retina_checker[ii].model(x_input.to(self.retina_checker[ii].device))
                pred.append(torch.nn.Sigmoid()(output).detach().cpu().numpy())
            prediction = np.array(pred).mean(0)
        return prediction