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
def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)

# Function to load checkpoint (including loss and accuracy lists)
def load_checkpoint(file_name='checkpoint.pth.tar'):
    checkpoint = torch.load(file_name)
    if 'state_dict' not in checkpoint or 'optimizer' not in checkpoint:
        raise KeyError("Checkpoint is missing required keys: 'state_dict' or 'optimizer'.")
    return checkpoint


# Smoothing function using a simple moving average
def smooth_curve(values, smoothing_factor=0.8):
    smoothed_values = []
    last_value = values[0]
    for value in values:
        smoothed_value = last_value * smoothing_factor + (1 - smoothing_factor) * value
        smoothed_values.append(smoothed_value)
        last_value = smoothed_value
    return smoothed_values

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, output_dir='plots', smoothing_factor=0.8):
    epochs = range(1, len(train_losses) + 1)

    # Create directory to save plots if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Smooth the curves
    train_losses_smoothed = smooth_curve(train_losses, smoothing_factor)
    val_losses_smoothed = smooth_curve(val_losses, smoothing_factor)
    train_accuracies_smoothed = smooth_curve(train_accuracies, smoothing_factor)
    val_accuracies_smoothed = smooth_curve(val_accuracies, smoothing_factor)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_smoothed, 'g', label='Training loss')
    plt.plot(epochs, val_losses_smoothed, 'b', label='Validation loss')
    plt.title('Training and Validation Loss (ResNet)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies_smoothed, 'g', label='Training accuracy')
    plt.plot(epochs, val_accuracies_smoothed, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy (ResNet)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save accuracy curve
    accuracy_plot_path = os.path.join(output_dir, 'ResNet_loss_accuracy_curve_smoothed.png')
    plt.savefig(accuracy_plot_path)
    print(f"Smoothed accuracy curve saved to {accuracy_plot_path}")

    # Show the plots (optional)
    plt.tight_layout()
    plt.close()


# Function to plot confusion matrix
def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (VGG19)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    # Save accuracy curve
    confusion_matrix_plot_path = os.path.join('plots', 'VGG19_confusion_matrix_curve.png')
    plt.savefig(confusion_matrix_plot_path)
    print(f"Confusion Matrix curve saved to {confusion_matrix_plot_path}")
    plt.close()
