import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# import os
# print("Current working directory:", os.getcwd())

"""
How do imports work?
Absolute import: 
Relative import: only works if we run the script as part of a package from top directory
"""
# ABSOLUTE IMPORTS
from nn.models.googlenet import create_googlenet
from nn.training.data_loader import get_data_loaders


def train():

    ########
    # DEVICE
    ########
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else device)
    print(f"Device: {device}")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")

    # else:
    #     print ("MPS device not found.")

    model = create_googlenet()
    val_data, val_loader = get_data_loaders()
    model = model.to(device)

    # Don't need to train!

    ########
    # Loss function and optimizer
    ########

    # criterion = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4) # SGD is common for ImageNet

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1) # Reduce the learning rate by a factor of 10 every 8 epochs

    # ########
    # # Training loop
    # ########

    # epochs = 100

    # for epoch in range(epochs):
    #     model.train()
    #     running_loss = 0
    #     correct = 0
    #     total = 0

    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()

    #         # Forward pass
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)

    #         # Backward pass
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item() * images.size(0)

    #         # Training accuracy
    #         _, predicted = torch.max(outputs, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #     epoch_loss = running_loss / len(train_loader.dataset)
    #     train_accuracy = correct / total * 100
    #     print(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}, Train accuracy: {train_accuracy:.2f}%')

    #     scheduler.step()

    #     ########
    #     # Validation loop
    #     ########

    #     # model.eval()
    #     # with torch.no_grad():
    #     #     for images, labels in val_loader:
    #     #         images, labels = images.to(device), labels.to(device)
    #     #         outputs = model(images)
    #     #         _, predicted = torch.max(outputs, 1) # Get the class index with the highest probability
    #     #         total += labels.size(0) # Add the number of labels in this batch
    #     #         correct += (predicted == labels).sum().item()
    #     # val_accuracy = correct / total * 100
    #     # print(f'Validation accuracy: {val_accuracy:.2f}%')

    ########
    # Evaluate
    ########

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    print(f"Validation accuracy: {val_accuracy:.2f}%")


if __name__ == "__main__":
    train()
