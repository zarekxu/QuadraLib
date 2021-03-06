from .vgg import *
# from .dpn import *
# from .lenet import *
# from .senet import *
# from .pnasnet import *
# from .densenet import *
# from .googlenet import *
# from .shufflenet import *
# from .shufflenetv2 import *
from .resnet import *
# from .resnext import *
# from .preact_resnet import *
from .mobilenet1 import *
# from .mobilenetv2 import *
# from .efficientnet import *
# from .regnet import *
# from .QVGG import *
# from .QVGG_SENET import *


def get_model(config):
    return globals()[config.architecture](config.num_classes)