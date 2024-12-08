import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
from glob import glob

class PretrainedFeatureExtractor:
    def __init__(self, model_name="resnet18", layer="avgpool", device="cpu"):
        self.device = device
        self.model = self._load_pretrained_model(model_name, layer)

    def _load_pretrained_model(self, model_name, layer):
        model = getattr(models, model_name)(pretrained=True)
        model = model.to(self.device)
        model.eval()
        self.layer = layer
        self.model_name = model_name
        return model

    def extract_features(self, image_folder, batch_size=16):
        """
        Extract features from a folder of images.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = ImageDataset(image_folder, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        features = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                features.append(outputs.cpu().numpy())

        features = np.vstack(features)
        return features

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = glob(os.path.join(image_folder, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
