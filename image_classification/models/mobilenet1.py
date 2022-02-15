'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
# from torchsummary import summary
import torch.nn.functional as F
from .quadratic_layer import Quadraour, Type1, Type2, Type3, Type4

# class Quadratic(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, groups, padding, bias):
#         super(Quadratic, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=stride, groups=groups, padding=padding, bias = True)
#         self.conv2 = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=stride, groups=groups, padding=padding, bias = True)
#         self.conv3 = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,stride=stride, groups=groups, padding=padding, bias = True)
#         # self.bn = nn.BatchNorm2d(out_channels)
        
#     def forward(self,x):
        
#         y0 = self.conv1(x)
#         y1 = self.conv2(x)
#         y2 = self.conv3(x)
#         y = torch.mul(y0,y1)
        
#         y = y+y2
#         # y = self.bn(y)
#         # y = F.relu(y)
#         return y


cfg = {
    "A": [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024],
    "B": [64, (128,2), 128, (256,2), 256, (512,2), 512, (1024,2)],
}




class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class QBlock(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(QBlock, self).__init__()
        self.conv1 = Quadraour(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, dilation = 1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = Quadraour(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation = 1, groups=1, bias=False)
        # self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    # cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]



    def __init__(self, cfg, num_classes=10, qua = False):
        super(MobileNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = Quadraour(3, 32, kernel_size=3, stride=1, padding=1, dilation = 1, groups=1, bias=False)
        # self.conv1 = Quadratic(3, 32,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(cfg, qua, in_planes=32,)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, cfg, qua, in_planes):
        layers = []

        if qua == False:
            print("start with first order mobilenet")
            for x in cfg: 
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(Block(in_planes, out_planes, stride))
                in_planes = out_planes
        elif qua == True:
            print("start with quadratic mobilenet")
            for x in cfg:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(QBlock(in_planes, out_planes, stride))
                in_planes = out_planes               

        return nn.Sequential(*layers)   


    # def _make_layers(self, in_planes, qua = False):
    #     layers = []
    #     i = 0
    #     for x in self.cfg:
            
    #         out_planes = x if isinstance(x, int) else x[0]
    #         stride = 1 if isinstance(x, int) else x[1]
    #         if x == 64:
    #             layers.append(QBlock(in_planes, out_planes, stride))
    #             in_planes = out_planes
    #         else:
    #             layers.append(Block(in_planes, out_planes, stride))
    #             in_planes = out_planes
    #         i = i+1
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()


def mobilenet(num_classes):
    return MobileNet(cfg["A"], num_classes, qua = False)

def qmobilenet(num_classes):
    return MobileNet(cfg["B"], num_classes, qua = True)

