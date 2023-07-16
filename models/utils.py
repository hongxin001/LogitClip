import torch
from torchvision.models import densenet121, resnet34
import numpy as np
import torch.nn as nn

def build_model(model_type, num_classes, device, args):
    if model_type == "resnet34":
        net = resnet34(num_classes=num_classes)
    elif model_type == "densenet":
        net = densenet121(pretrained=False, num_classes=num_classes)
    net.to(device)
    return net