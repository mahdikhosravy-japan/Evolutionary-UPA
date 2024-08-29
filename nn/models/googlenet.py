
import torch
import torch.nn as nn
import torchvision.models as models

def create_googlenet():

    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT) # Load the pretrained weights

    # Change the number of output classes (default is 1000, but ImageNet is 1024)
    model.fc = nn.Linear(1024, 1000)
    return model