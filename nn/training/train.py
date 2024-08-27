import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


########
# Data transformations
########

# Normalize the image to the ImageNet mean and standard deviation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

########
# Data loading
########

# Load the data
train_data = datasets.ImageFolder('../data/ILSVRC2012/val', transform=train_transforms)
val_data = datasets.ImageFolder('../data/ILSVRC2012/val', transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)