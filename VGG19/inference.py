import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

IMG_SIZE = (768, 768)
NUM_CLASSES = 4  # Number of classes
CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']  # Example class names

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)

# Function to perform inference with confidence score
def inference(image_path):
    # Load the VGG19 model
    model = models.vgg19(pretrained=False)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, NUM_CLASSES)
    
    model_path = '/home/bapary/Work/Classification/VGG19/models/vgg19_checkpoint.pth.tar'

    # Load the trained model weights and map to CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"])
    model.eval()  # Set model to evaluation mode

    # Preprocess the image
    image = preprocess_image(image_path)
    image = image.to('cpu')  # Ensure the image tensor is on the CPU

    # Perform inference
    with torch.no_grad():
        output = model(image)

        # Calculate probabilities using softmax
        probabilities = F.softmax(output, dim=1)

        # Get the predicted class and the associated confidence score
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASSES[predicted.item()]
    confidence_score = confidence.item() * 100  # Convert to percentage

    print(f'Predicted class: {predicted_class}')
    print(f'Confidence score: {confidence_score:.2f}%')

if __name__ == "__main__":
    test_image_path = '/home/bapary/Work/Classification/Downloaded_image/download (1).jpeg'
    inference(test_image_path)
