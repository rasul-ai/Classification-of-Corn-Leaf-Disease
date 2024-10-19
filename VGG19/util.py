import os
import cv2
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item()

# Function to save checkpoint (including loss and accuracy lists)
def save_checkpoint(state, file_name='vgg19_checkpoint.pth.tar'):
    torch.save(state, file_name)

# Function to load checkpoint (including loss and accuracy lists)
def load_checkpoint(file_name='vgg19_checkpoint.pth.tar'):
    checkpoint = torch.load(file_name)
    if 'state_dict' not in checkpoint or 'optimizer' not in checkpoint:
        raise KeyError("Checkpoint is missing required keys: 'state_dict' or 'optimizer'.")
    return checkpoint

# Function to plot training/validation loss and accuracy
def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, output_dir='plots'):
    epochs = range(1, len(train_losses) + 1)

    # Create directory to save plots if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation Loss (VGG19)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'g', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy (VGG19)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save accuracy curve
    accuracy_plot_path = os.path.join(output_dir, 'VGG19_loss_accuracy_curve.png')
    plt.savefig(accuracy_plot_path)
    print(f"Accuracy curve saved to {accuracy_plot_path}")

    # Show the plots (optional)
    plt.tight_layout()
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (ResNet)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    # Save accuracy curve
    confusion_matrix_plot_path = os.path.join('plots', 'ResNet_confusion_matrix_curve.png')
    plt.savefig(confusion_matrix_plot_path)
    print(f"Confusion Matrix curve saved to {confusion_matrix_plot_path}")
    plt.show()
