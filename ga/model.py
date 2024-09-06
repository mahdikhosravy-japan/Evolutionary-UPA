
import torch
import torchvision.models as models
import torch.nn.functional as F

# from nn.models.googlenet import create_googlenet

# model = create_googlenet()

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

