# from .vgg16 import Vgg16
# from .alexnet import AlexNet
# from .vgg16_112 import Vgg16_112
# from .vgg16_112_7x7 import Vgg16_112_7x7
#
# def get_model(num_classes, model_type, init_block_num=0, seed=0):
#     if model_type == 'vgg16':
#         model = Vgg16(num_classes=num_classes, init_block_num=init_block_num, seed=seed)
#     elif model_type == 'vgg16_112' or model_type == 'vgg16_96':
#         model = Vgg16_112(num_classes=num_classes, init_block_num=init_block_num, seed=seed)
#     elif model_type == 'vgg16_112_7x7':
#         model = Vgg16_112_7x7(num_classes=num_classes, init_block_num=init_block_num, seed=seed)
#     elif model_type == 'alexnet':
#         model = AlexNet(num_classes=num_classes, init_block_num=init_block_num, seed=seed)
#     else:
#         raise Exception('Invalid model_type')
#     return model
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models import alexnet
from torchvision.models import resnet18, resnet34, resnet50


def get_model(num_classes, model_type):
    if model_type == 'vgg16':
        model = vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    elif model_type == 'alexnet':
        model = alexnet(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    elif model_type == 'res18':
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
    elif model_type == 'res34':
        model = resnet34(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
    elif model_type == 'res50':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    else:
        raise Exception('Invalid model_type')
    return model


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def forward(self, x):
        return (x - self.mean.to(x.device)[None, :, None, None]) / self.std.to(
            x.device)[None, :, None, None]
