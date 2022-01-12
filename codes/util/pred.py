import torch
import numpy as np


def predict(image, rgb_mean, rgb_std, device, model):
    assert image.shape[1] == 3 and rgb_std.shape[0] == 3
    img = preprocess(image, rgb_mean, rgb_std)
    img_tensor = torch.from_numpy(img).to(device)
    with torch.no_grad():
        out = model(img_tensor).data.cpu().numpy().squeeze()
    return out


def preprocess(image, mean, std):
    img = (image - mean) / std
    return img.astype(np.float32)
