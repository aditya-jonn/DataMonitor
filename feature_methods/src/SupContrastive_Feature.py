import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from glob import glob
import os

class SupContrastiveLearningModel(nn.Module):
    """
    Supervised Contrastive Learning Model using a ResNet backbone.
    """
    def __init__(self, feature_dim=128, device="cuda"):
        super(SupContrastiveLearningModel, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
        self.projection_head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        self.to(self.device)

    def forward(self, x):
        features = self.encoder(x).squeeze()  # Global Average Pooling
        projections = self.projection_head(features)
        return projections

class SupContrastiveDataset(Dataset):
    """
    Dataset for Supervised Contrastive Learning.
    """
    def __init__(self, image_folder, labels, transform=None):
        self.image_paths = glob(os.path.join(image_folder, "*.jpg"))
        self.labels = labels  # Dictionary mapping image filenames to labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[os.path.basename(image_path)]
        if self.transform:
            image = self.transform(image)
        return image, label

class SupContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    """
    def __init__(self, temperature=0.07):
        super(SupContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Compute the contrastive loss.
        :param features: NxD tensor, where N is batch size and D is feature dimension.
        :param labels: N tensor, where N is batch size.
        :return: Loss value.
        """
        batch_size = features.size(0)
        features = nn.functional.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        labels_matrix = labels_matrix.float().to(features.device)

        # Mask out self-comparisons
        mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)
        labels_matrix = labels_matrix * (~mask)

        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        numerator = exp_sim * labels_matrix
        denominator = exp_sim.sum(dim=1, keepdim=True) - exp_sim * mask

        loss = -torch.log(numerator.sum(dim=1) / denominator.sum(dim=1) + 1e-8).mean()
        return loss
