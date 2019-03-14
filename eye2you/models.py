from torchvision.models import (alexnet, densenet121, densenet161, densenet169, densenet201, inception_v3, resnet18,
                                resnet34, resnet50, resnet101, resnet152, squeezenet1_0, squeezenet1_1, vgg11, vgg11_bn,
                                vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn)

from .inception_short import inception_v3_s, inception_v3_xs

__models__ = [
    alexnet, densenet121, densenet161, densenet169, densenet201, inception_v3, resnet18, resnet34, resnet50, resnet101,
    resnet152, squeezenet1_0, squeezenet1_1, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
    inception_v3_s, inception_v3_xs
]
