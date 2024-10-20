import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Define the dataset class
class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.images = []
        self.labels = []

        # Load image paths and labels
        for label, folder in enumerate(classes):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.images.append(os.path.join(folder_path, file))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Data transformations
def get_data_transforms(img_size=(768, 768)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
