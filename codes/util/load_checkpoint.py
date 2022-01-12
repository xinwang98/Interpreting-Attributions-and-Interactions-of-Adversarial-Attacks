import torch
import torch.nn as nn
from codes.util.get_model import get_model, Normalize


def load_checkpoint(num_classes, model_type, checkpoint_path, device, train_type='', pretrain='pretrained'):
    model = get_model(num_classes=num_classes, model_type=model_type)
    check_point = torch.load(checkpoint_path, map_location=device)

    if 'bin' in checkpoint_path:
        model.load_state_dict(check_point)
    else:
        model.load_state_dict(check_point['state_dict'])
        acc = check_point['acc']
        print('acc of {} is {}'.format(model_type, acc))
    model.eval()
    return model

