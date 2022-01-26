'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torchvision

import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + ".pth.tar")
    if is_best:
        shutil.copyfile(filename + ".pth.tar", filename + "_best.pth.tar")


def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        logging.info("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if optimizer is not None:
            best_prec = checkpoint["best_prec"]
            last_epoch = checkpoint["last_epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                "=== done. also loaded optimizer from "
                + "checkpoint '{}' (epoch {}) ===".format(path, last_epoch + 1)
            )
            return best_prec, last_epoch


def get_data_loader(transform_train, transform_test, config):
    assert config.dataset == "cifar10" or config.dataset == "cifar100"
    if config.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=True, download=True, transform=transform_train
        )

        testset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=False, download=True, transform=transform_test
        )
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=True, download=True, transform=transform_train
        )

        testset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=False, download=True, transform=transform_test
        )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch, shuffle=False, num_workers=config.workers
    )
    return train_loader, test_loader



def data_augmentation(config, is_train=True):
    aug = []
    if is_train:
        # random crop
        if config.augmentation.random_crop:
            aug.append(transforms.RandomCrop(config.input_size, padding=4))
        # horizontal filp
        if config.augmentation.random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())

    aug.append(transforms.ToTensor())
    # normalize  [- mean / std]
    if config.augmentation.normalize:
        if config.dataset == "cifar10":
            aug.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            )
        else:
            aug.append(
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            )

    if is_train and config.augmentation.cutout:
        # cutout
        aug.append(
            Cutout(n_holes=config.augmentation.holes, length=config.augmentation.length)
        )
    return aug