import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import yaml

from PIL import Image


########
# CONFIG
########
def load_config(config_file="ga/config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

########
# PREPROCESSING
########

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def preprocess_image_batch(batch_size, image_dir):
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) # Normalize the image to the ImageNet mean and standard deviation


    # Load and apply preprocessing
    dataset = datasets.ImageFolder(image_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  

    # Get 1 batch
    batch = next(iter(dataloader))

    return batch[0], batch[1] # Return image and labels


########
# VISUALIZATION
########

# Convert a tensor to an image
def denormalize_image(tensor):
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std_tensor + mean_tensor
    return tensor

def visualize_perturbation(input_image, perturbation):
    perturbed_image = input_image + perturbation
    # perturbed_image = torch.clamp(perturbed_image, 0, 1) # Ensure the pixel values are between 0 and 1
    input_image_denormalized = denormalize_image(input_image).squeeze(0) # squeeze here removes the batch dimension
    perturbed_image_denormalized = denormalize_image(perturbed_image).squeeze(0)

    # Convert tensor to numpy for visualization
    input_image_np = input_image_denormalized.permute(1, 2, 0).cpu().numpy() # HWC
    perturbed_image_np = perturbed_image_denormalized.permute(1, 2, 0).cpu().numpy() # HWC
    perturbation_np = perturbation.squeeze(0).permute(1, 2, 0).cpu().numpy() # HWC

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(input_image_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    axs[1].imshow(np.clip(perturbed_image_np, 0, 1))
    axs[1].set_title("Perturbed Image")
    axs[1].axis("off")
    axs[2].imshow(np.clip(perturbation_np, 0, 1))
    axs[2].set_title("Perturbation")
    axs[2].axis("off")

    plt.show()