# -*-coding:utf-8-*-
import torch.nn as nn
from .quadratic_layer import Quadraour, Type1, Type2, Type3, Type4

__all__ = ["vgg11", "vgg13", "vgg16", "vgg19", "qvgg7"]

cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    "Q7": [64, 64, "M", 128, 128, "M", 256, "M", 512, "M", 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, qua=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm = False, qua = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            if qua == False:
                conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
            elif qua == True:
                conv2d = Quadraour(in_channels, v, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11(num_classes):
    return VGG(make_layers(cfg["A"], batch_norm=True, qua = False), num_classes)


def vgg13(num_classes):
    return VGG(make_layers(cfg["B"], batch_norm=True, qua = False), num_classes)


def vgg16(num_classes):
    return VGG(make_layers(cfg["D"], batch_norm=True, qua = False), num_classes)


def vgg19(num_classes):
    return VGG(make_layers(cfg["E"], batch_norm=True, qua = False), num_classes)


def qvgg7(num_classes):
    return VGG(make_layers(cfg["Q7"], batch_norm=True, qua = True), num_classes)

def qvgg13(num_classes):
    return VGG(make_layers(cfg["D"], batch_norm=True, qua = True), num_classes)