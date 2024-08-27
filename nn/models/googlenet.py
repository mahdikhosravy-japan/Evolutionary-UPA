
import torch
import torch.nn as nn
import torchvision.models as models

model = models.googlenet(pretrained=False)

# Change the number of output classes (default is 1000, but ImageNet is 1024)
model.fc = nn.Linear(1024, 1000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)