
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    """
    Returns a CNN model for pneumonia classification
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
