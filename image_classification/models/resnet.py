'''ResNet in PyTorch.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quadratic_layer import Quadraour, Type1, Type2, Type3, Type4



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, i=0, stride=1, qua = False):
        super(BasicBlock, self).__init__()

        print("iii is:", i)

        if qua == True:
            # print("execution")
            self.conv1 = Quadraour(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, dilation = 1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = Quadraour(planes, planes, 3,  1, padding = 1, dilation = 1)
            # self.conv2 = nn.conv2d(planes, planes, kernel_size=3,
                                   # stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    # Quadratic1(in_planes, self.expansion*planes, stride),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        else: 
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10, qua = False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # self.conv1 = Quadratic(3, 64, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, qua = qua)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, qua = qua)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, qua = qua)
        # self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2, quadra=True)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, qua):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            i = i+1
            print("i is:", i)
            layers.append(block(self.in_planes, planes, i, stride, qua))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




def qresnet10(num_classes):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes, qua = True)

def qresnet14(num_classes):
    return ResNet(BasicBlock, [2, 2, 2], num_classes, qua = True)

def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, qua = False)

def resnet32(num_classes):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, qua = False)

def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, qua = False)


def resnet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, qua = False)


def resnet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, qua = False)


def resnet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, qua = False)


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
    # print(y.size())

# test()
