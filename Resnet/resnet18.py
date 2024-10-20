import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from util import plot_confusion_matrix, plot_loss_accuracy, calculate_accuracy, save_checkpoint, load_checkpoint
from config import DATASET_PATH, IMG_SIZE, BATCH_SIZE, NUM_CLASSES, CLASSES

# Main training function
def train(resume=False):
    # Data transformations with more aggressive augmentations
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(30),  # Increased rotation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # More aggressive augmentation
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, 'val'), transform=transform)

    # Create DataLoaders with a smaller batch size to reduce overfitting
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the ResNet18 model (reduced complexity)
    model = models.resnet18(pretrained=True)  # Switched to ResNet18 to reduce complexity
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Added Dropout
        nn.Linear(num_features, NUM_CLASSES)
    )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)  # Increased weight decay

    # Learning rate scheduler with CosineAnnealingLR for better learning rate adjustments
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Variables to track losses and accuracies
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # For resuming training
    start_epoch = 1
    if resume:
        if os.path.exists("resnet_checkpoint_test_3.pth.tar"):
            checkpoint = load_checkpoint("resnet_checkpoint_test_3.pth.tar")
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

    # Early stopping variables
    patience = 20
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    num_epochs = 82
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Progress bar for training
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training]", unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Applied gradient clipping

                optimizer.step()

                running_loss += loss.item()
                correct_train += calculate_accuracy(outputs, labels)
                total_train += labels.size(0)

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
            with tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Validation]", unit="batch") as vepoch:
                for images, labels in vepoch:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    running_val_loss += loss.item()
                    correct_val += calculate_accuracy(outputs, labels)
                    total_val += labels.size(0)

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    vepoch.set_postfix(loss=running_val_loss / len(val_loader), accuracy=correct_val / total_val)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        # Step the learning rate scheduler
        scheduler.step()

        # Save metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print epoch results with time taken
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Time Taken: {epoch_time:.2f}s")

        # Save latest checkpoint
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies
        }
        save_checkpoint(checkpoint, file_name="resnet_checkpoint_test_4.pth.tar")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

    # Plot loss and accuracy curves
    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)

    # Plot confusion matrix after the last epoch
    plot_confusion_matrix(all_labels, all_preds, CLASSES)

    print("Training completed.")


if __name__ == "__main__":
    # To resume training, set resume=True
    train(resume=True)
