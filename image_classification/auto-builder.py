import argparse
import time
import os
import numpy as np
import yaml


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models import get_model
from utils import (
    progress_bar,
    data_augmentation,
    get_data_loader,
    load_checkpoint,
)


from easydict import EasyDict


os.environ["CUDA_VISIBLE_DEVICES"]="0"




parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()

# read config from yaml file
with open(args.work_path + "/config_auto_builder.yaml") as f:
    config = yaml.safe_load(f)
# convert to dict
config = EasyDict(config)
print(type(config.layer))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
# load training data, do data augmentation and get data loader
transform_train = transforms.Compose(data_augmentation(config))

transform_test = transforms.Compose(data_augmentation(config, is_train=False))

trainloader, testloader = get_data_loader(transform_train, transform_test, config)


# Model
print('==> Building model..')

net = get_model(config)

print(net)

net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



criterion = nn.CrossEntropyLoss()


def masking(weight, p):
    weight = F.dropout(weight, p = p) 
    return weight

def test(epoch, nlayer, p):
    global best_acc
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/qvgg13_cifar10.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    print("layer number is:", nlayer)
    
    net.eval()


    net.features[nlayer].conv1.weight.data = masking(net.features[nlayer].conv1.weight.data, p)
    net.features[nlayer].conv2.weight.data = masking(net.features[nlayer].conv2.weight.data, p)
    net.features[nlayer].conv3.weight.data = masking(net.features[nlayer].conv3.weight.data, p)


    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        print(type(net.features[nlayer].conv1.weight.data))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            imgs = inputs.data.cpu().numpy()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



for epoch in range(0, config.layernumber):
    

    test(epoch, config.layer_index[epoch], config.masking_ratio)
