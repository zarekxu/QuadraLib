from model import SSD300
from utils import adjust_learning_rate
import torch

def get_model(config):
    return globals()[config.architecture](config.num_classes)

def get_optim(model, config):
    if config.optim.optimizer == 'SGD':
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        return torch.optim.SGD(params=[{'params': biases, 'lr': 2 * config.learning_rate}, {'params': not_biases}],
                                    lr=config.learning_rate, momentum=config.optim.momentum, weight_decay=float(config.optim.weight_decay))

def update_optim(optimizer, epoch, config):
    if config.optim.optimizer == 'SGD':
        if epoch in config.optim.decay_at:
            adjust_learning_rate(optimizer, config.optim.decay_rate, config.final_learning_rate)