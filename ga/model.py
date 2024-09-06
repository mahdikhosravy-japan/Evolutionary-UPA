
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


# from nn.models.googlenet import create_googlenet

# model = create_googlenet()

########
# MODEL
########

# Load a pre-trained model
def load_model(model_type="googlenet"):
    if model_type == "googlenet":
        model = models.googlenet(pretrained=True)
    elif model_type == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    
    model.eval() # Set the model to evaluation mode
    return model


########
# PREPROCESSING
########

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def preprocess_image(image_path):
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) # Normalize the image to the ImageNet mean and standard deviation

    # Load and apply preprocessing
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0) # Add a batch dimension: 1 x 3 x 224 x 224

    return input_tensor

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


########
# PREDICTION
########

# Predict the class of an image
def predict_class(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    
    dist = F.softmax(output, dim=1) # Convert the output to a probability distribution
    predicted_class = torch.argmax(dist, dim=1).item()
    return predicted_class

def predict_with_perturbation(model, input_tensor, preturbation):
    perturbed_input = input_tensor + preturbation
    perturbed_input = torch.clamp(perturbed_input, 0, 1) # Ensure the pixel values are between 0 and 1
    return predict_class(model, perturbed_input)

