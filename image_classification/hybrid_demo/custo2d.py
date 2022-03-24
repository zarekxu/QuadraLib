import math
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.distributed as TUDdistributed
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import os
import sys
from copy import deepcopy
from torch.nn.parameter import Parameter



class MyConv2d_bp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        h_in, w_in = input.shape[2:]
        h_out = h_in
        w_out = w_in
        kernel_size = weight.shape[2]
        in_channels = input.shape[1]
        # x: [batchsize ksize num_sliding]
        x = torch.nn.functional.unfold(input, kernel_size=3, padding=1)
        # print("xxx is:", x.size())

        batchsize = input.shape[0]
        ksize = in_channels*kernel_size*kernel_size
        num_sliding = x.shape[2]

        assert x.shape[1] == ksize

        w = self.weight
        # print("aaaaaa", x.size())
        out_unf = x.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # print("bbbbbbb", out_unf.size())
        out = torch.nn.functional.fold(out_unf, output_size=[h_out, w_out], kernel_size=1, padding=0, dilation=1, stride=1)

        return out


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, weight = ctx.saved_tensors



        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class Myfc_bpp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        # print("input size is:", input.size())
        # print("weight size is:", weight.size())

        in_n = input.shape[1]
        out_n = weight.shape[1]
        out = input.matmul(weight)
        # out = x.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        return out


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, weight = ctx.saved_tensors
        # print("grad_output size :", grad_output.size())
        grad_input = torch.mm(grad_output, weight.transpose(0,1))
        grad_weight = torch.mm(input.transpose(0,1), grad_output)
        return grad_input, grad_weight


class Myqfc_bpp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, weight1, weight2):
        ctx.save_for_backward(input, weight1, weight2)
        # print("input size is:", input.size())
        # print("weight size is:", weight.size())

        in_n = input.shape[1]
        out_n = weight1.shape[1]
        out1 = input.matmul(weight1)
        out2 = input.matmul(weight2)
        out = torch.mul(out1, out2)
        out = out

        # out = x.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        return out


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, weight1, weight2 = ctx.saved_tensors
        # print("grad_output size :", grad_output.size())
        grad_input = torch.mm(torch.mul(grad_output, input.matmul(weight1)), weight2.transpose(0,1)) + torch.mm(torch.mul(grad_output, input.matmul(weight2)), weight1.transpose(0,1))

        # grad_input = torch.mm(grad_output, weight.transpose(0,1))																								
        grad_weight1 = torch.mm(input.transpose(0,1), torch.mul(grad_output, input.matmul(weight2)))																																																																																																	
        grad_weight2 = torch.mm(input.transpose(0,1), torch.mul(grad_output, input.matmul(weight1)))					
        return grad_input, grad_weight1, grad_weight2



# class MyConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
#         super(MyConv2d, self).__init__()
#         self.conv = Myconv2D.apply
#         self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
#         

class Myfc_bp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Myfc_bp, self).__init__()
        # self.fc = Myfc_bp.apply
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
    def forward(self, x):
        return Myfc_bpp.apply(x, self.weight)


class Myqfc_bp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Myqfc_bp, self).__init__()
        # self.fc = Myfc_bp.apply
        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight3 = Parameter(torch.Tensor(in_channels, out_channels))
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        out = Myqfc_bpp.apply(x, self.weight1, self.weight2)
        out = out + self.fc(x)
        return out

class Myqfc_bp1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Myqfc_bp1, self).__init__()
        self.fc1 = nn.Linear(128, 10)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
    	out = torch.mul(self.fc1(x), self.fc2(x))
    	out = out + self.fc3(x)
    	return out