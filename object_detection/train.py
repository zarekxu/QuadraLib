import yaml
import argparse
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import MultiBoxLoss
from dataloader import get_data_loader
from utils import *

from models import get_model, get_optim, update_optim

from easydict import EasyDict

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
# parser.add_argument('--test', '-t', action='store_true', help='Test only flag')

args = parser.parse_args()

with open(args.work_path + "/config.yaml") as f:
    config = yaml.safe_load(f)
config = EasyDict(config)

config_name = config.architecture + '_' + config.dataset
print('==> config name: ', config_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print_freq = 200  # print training status every __ batches
# Learning parameters

cudnn.benchmark = True

def main(config):
    """
    Training.
    """
    global start_epoch, label_map, checkpoint

    # Initialize model or load checkpoint
    if not args.resume:
        start_epoch = 0

        model = get_model(config)
        optimizer = get_optim(model, config)

    else:
        ckpt_path = os.path.join('./checkpoint/' + config_name + '.pth')
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Data Loader
    train_loader, test_loader = get_data_loader(config)

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations

    # Epochs
    for epoch in range(start_epoch, epoch_from(config)):

        # Decay learning rate at particular epochs
        update_optim(optimizer, epoch, config)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        validate()

        # Save checkpoint
        save_checkpoint(os.path.join('./checkpoint', config_name + '.pth'),
                        epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if config.grad_clip != "None":
            clip_gradient(optimizer, config.grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

def validate():
    print("NOT IMPLEMENTED YET")

if __name__ == '__main__':
    main(config)
