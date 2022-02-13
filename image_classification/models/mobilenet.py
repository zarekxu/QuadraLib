# https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV1.py

import torch
import torch.nn as nn
from torchsummary import summary
from .quadratic_layer import Quadraour, Type1, Type2, Type3, Type4
class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

class QMobileNet(MobileNetV1):
    def __init__(self, ch_in, num_classes):
        super(QMobileNet, self).__init__(3, num_classes)

        def conv_bn_qua(inp, oup, stride):
            return nn.Sequential(
                Quadraour(inp, oup, 3, stride, 1, 1),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw_qua(inp, oup, stride):
            return nn.Sequential(
                # dw
                Quadraour(inp, inp, 3, stride, 1, 1, groups=inp),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn_qua(ch_in, 32, 2),
            conv_dw_qua(32, 64, 1),
            conv_dw_qua(64, 128, 2),
            conv_dw_qua(128, 128, 1),
            conv_dw_qua(128, 256, 2),
            conv_dw_qua(256, 256, 1),
            conv_dw_qua(256, 512, 2),
            conv_dw_qua(512, 512, 1),
            conv_dw_qua(512, 512, 1),
            conv_dw_qua(512, 512, 1),
            conv_dw_qua(512, 512, 1),
            conv_dw_qua(512, 512, 1),
            conv_dw_qua(512, 1024, 2),
            conv_dw_qua(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )


def qmobilenet13(num_classes):
    return QMobileNet(3, num_classes)

def mobilenet13(num_classes):
    return MobileNetV1(3, num_classes)

if __name__=='__main__':
    net = qmobilenet13(10)
    summary(net, (3, 32, 32), 256)
