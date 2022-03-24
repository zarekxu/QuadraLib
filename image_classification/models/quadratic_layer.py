import torch.nn as nn
import torch.nn.functional as F
import torch

class Quadraour(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
		super(Quadraour, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size , stride, padding, dilation, groups, bias)
		self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size , stride, padding, dilation, groups, bias)
		self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size , stride, padding, dilation, groups, bias)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.bn3 = nn.BatchNorm2d(out_channels)
        
	def forward(self,x):
        
		y0 = self.conv1(x)
		y0 = self.bn1(y0)
		y1 = self.conv2(x)
		y1 = self.bn2(y1)
		y2 = self.conv3(x)
		y2 = self.bn3(y2)
		y = torch.mul(y0,y1)
		y = y + y2
		return y


class Type1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super(Type1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.bilconv = nn.Bilinear(kernel_size*kernel_size*in_channels, kernel_size*kernel_size*in_channels, out_channels, bias = bias)



    def forward(self, input):
        h_in, w_in = input.shape[2:]
        h_out = h_in
        w_out = w_in
        x = torch.nn.functional.unfold(input, kernel_size=self.kernel_size, padding=self.padding)

        batchsize = input.shape[0]
        ksize = self.in_channels*self.kernel_size*self.kernel_size
        num_sliding = x.shape[2]

        assert x.shape[1] == ksize

        x=x.view(x.size(0),x.size(2),x.size(1))
        out_unf = self.bilconv(x,x)
        out_unf=out_unf.view(out_unf.size(0),out_unf.size(2),out_unf.size(1))
        out = torch.nn.functional.fold(out_unf, output_size=[h_out, w_out], kernel_size=1, padding=0, dilation=self.dilation, stride=1)

        return out


class Type3(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
		super(Type3, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        
	def forward(self,x):
		print("aaaaa", x.size())
		y = self.conv1(x)
		return y*y



class Type2(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
		super(Type2, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
	def forward(self,x):
        
		y = self.conv1(x*x)
		return y


class Type4(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
		super(Type4, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
	def forward(self,x):
        
		y0 = self.conv1(x)
		y1 = self.conv2(x)
		y = torch.mul(y0,y1)
		return y



class Quadrafc(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
		super(Quadraour, self).__init__()
		self.fc1 = nn.Linear(in_channels, out_channels)
		self.fc2 = nn.Linear(in_channels, out_channels)
		self.fc3 = nn.Linear(in_channels, out_channels)

        
	def forward(self,x):
        
		y0 = self.fc1(x)
		y1 =  self.fc2(x)
		y2 =  self.fc3(x)
		y = torch.mul(y0,y1)
		y = y + y2
		return y


class qfc_bp(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input, weight1, weight2):
        ctx.save_for_backward(input, weight1, weight2)

        in_n = input.shape[1]
        out_n = weight1.shape[1]
        out1 = input.matmul(weight1)
        out2 = input.matmul(weight2)
        out = torch.mul(out1, out2)
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
        grad_weight1 = torch.mm(input.transpose(0,1), torch.mul(grad_output, input.matmul(weight2)))																																																																																																	
        grad_weight2 = torch.mm(input.transpose(0,1), torch.mul(grad_output, input.matmul(weight1)))					
        return grad_input, grad_weight1, grad_weight2



class Quadrafc_hybrid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Quadrafc_hybrid, self).__init__()
        # self.fc = Myfc_bp.apply
        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight3 = Parameter(torch.Tensor(in_channels, out_channels))
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        out = qfc_bp.apply(x, self.weight1, self.weight2)
        out = out + self.fc(x)
        return out