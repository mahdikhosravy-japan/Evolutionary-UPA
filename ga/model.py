
import torch
import torch.nn.functional as F
import torchvision.models as models


# from nn.models.googlenet import create_googlenet

# model = create_googlenet()

########
# MODEL
########

# Load a pre-trained model
def load_model(model_type="googlenet"):
    if model_type == "googlenet":
        model = models.googlenet(weights='DEFAULT')
    elif model_type == "resnet18":
        model = models.resnet18(weights='DEFAULT')
    elif model_type == "resnet50":
        model = models.resnet50(weights='DEFAULT')
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    
    model.eval() # Set the model to evaluation mode
    return model


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

