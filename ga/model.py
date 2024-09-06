
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

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

def preprocess_image(image_path):
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) # Normalize the image to the ImageNet mean and standard deviation

    # Load and apply preprocessing
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0) # Add a batch dimension: 1 x 3 x 224 x 224

    return input_tensor



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

