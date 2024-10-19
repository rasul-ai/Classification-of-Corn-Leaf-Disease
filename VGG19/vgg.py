import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import itertools
from util import plot_confusion_matrix, plot_loss_accuracy, calculate_accuracy, save_checkpoint, load_checkpoint
from config import DATASET_PATH, IMG_SIZE, BATCH_SIZE, NUM_CLASSES, CLASSES

# Main training function
def train(resume=False):
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(20),  # Reduced rotation for less noise
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, 'val'), transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the VGG19 model pre-trained on ImageNet
    model = models.vgg19(pretrained=True)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, NUM_CLASSES)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)  # Reduced learning rate, added weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Variables to track losses and accuracies
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # For resuming training
    start_epoch = 1
    if resume:
        if os.path.exists("vgg19_checkpoint_test_2.pth.tar"):
            checkpoint = load_checkpoint("vgg19_checkpoint_test_2.pth.tar")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            train_accuracies = checkpoint.get("train_accuracies", [])
            val_accuracies = checkpoint.get("val_accuracies", [])
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting fresh training")

    # Training loop
    num_epochs = 201  # Adjust as needed
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()  # Start timing the epoch

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Progress bar for training
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training]", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Track loss and accuracy
                running_loss += loss.item()
                correct_train += calculate_accuracy(outputs, labels)
                total_train += labels.size(0)

                # Update progress bar
                tepoch.set_postfix(loss=running_loss / len(train_loader), accuracy=correct_train / total_train)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            # Progress bar for validation
            with tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Validation]", unit="batch") as vepoch:
                for images, labels in vepoch:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Track validation loss and accuracy
                    running_val_loss += loss.item()
                    correct_val += calculate_accuracy(outputs, labels)
                    total_val += labels.size(0)

                    # Collect predictions and true labels for confusion matrix
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    # Update progress bar
                    vepoch.set_postfix(loss=running_val_loss / len(val_loader), accuracy=correct_val / total_val)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        # Step the learning rate scheduler based on validation loss
        scheduler.step(val_loss)

        # Save metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print epoch results with time taken
        epoch_time = time.time() - epoch_start_time  # Time taken for the epoch
        print(f"Epoch [{epoch}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Time Taken: {epoch_time:.2f}s")

        # Save latest checkpoint with all the data
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }
        save_checkpoint(checkpoint, file_name="vgg19_checkpoint_test_3.pth.tar")

    # Plot loss and accuracy curves
    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)

    # Plot confusion matrix after the last epoch
    plot_confusion_matrix(all_labels, all_preds, CLASSES)

    print("Training completed.")


if __name__ == "__main__":
    # To resume training, set resume=True
    train(resume=True)
