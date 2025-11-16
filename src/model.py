import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
