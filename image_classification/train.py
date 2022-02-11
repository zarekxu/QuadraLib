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


from torch.utils.tensorboard import SummaryWriter
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
with open(args.work_path + "/config.yaml") as f:
    config = yaml.safe_load(f)
# convert to dict
config = EasyDict(config)

config_name = config.architecture + '_' + config.dataset
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
# net = VGG('VGG19')
# net = vgg()
net = get_model(config)

print(net)

net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + config_name + '.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=config.learning_rate,
                      momentum=0.9, weight_decay=5e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.epoch, eta_min = 0.00001, last_epoch = -1)

writer = SummaryWriter('runs/' + config_name)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    for param_group in optimizer.param_groups:
        print('learning rate: %f' % param_group['lr'])
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        writer.add_scalar('loss value',loss, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, "./checkpoint/" + config_name + ".pth")
        best_acc = acc





if __name__=="__main__":
    for epoch in range(start_epoch, start_epoch+config.epoch):
        # a = time.time()
        train(epoch)
        scheduler.step()
        # print("training time is:", time.time()-a)
        test(epoch)
