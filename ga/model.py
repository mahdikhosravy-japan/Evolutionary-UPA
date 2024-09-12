
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

# # Predict the class of an image
# def predict(model, input_tensor):
#     with torch.no_grad():
#         output = model(input_tensor)
    
#     dist = F.softmax(output, dim=1) # Convert the output to a probability distribution
#     predicted_label = torch.argmax(dist, dim=1).item()
#     return predicted_label

def predict_batch(model, input_batch):
    with torch.no_grad():
        outputs = model(input_batch)
    
    predicted_labels = outputs.argmax(dim=1)
    return predicted_labels

def predict_with_perturbation(model, input_batch, preturbation):
    perturbed_input = input_batch + preturbation
    perturbed_input = torch.clamp(perturbed_input, 0, 1) # Ensure the pixel values are between 0 and 1
    return predict_batch(model, perturbed_input)


########
# EVALUATION
########

def evaluate_without_perturbation(model, dataloader):
    # Evaluate the model without the universal perturbation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy without perturbation: {accuracy}")
    return accuracy

def evaluate_with_perturbation(model, dataloader, perturbation):
    # Evaluate the model with the universal perturbation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            perturbed_images = images + perturbation
            perturbed_images = torch.clamp(perturbed_images, 0, 1) # Ensure the pixel values are between 0 and 1
            outputs = model(perturbed_images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy with perturbation: {accuracy}")
    return accuracy

