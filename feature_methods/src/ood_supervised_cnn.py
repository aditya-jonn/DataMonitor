"""OOD-supervised CNN feature extractor — faithful to the author's train_cnn.py.

Recipe (matches the author's contrastive-ood scripts):
  - torchvision resnet18 (random init unless --pretrained), 3-channel input
    (grayscale tripled to 3 channels); fc replaced with Linear(512, 2) + Sigmoid
  - BCELoss on one-hot targets; best checkpoint selected by validation AUROC
  - OOD features = the 512-d PENULTIMATE vector (the input to fc), matching
    EvaluateCNN's forward hook. NOT the 2-d logits — Mahalanobis needs the
    full-rank penultimate space, and cosine separates there once the net is
    trained 3-channel with a BCE+Sigmoid head.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .models.base import Model, FeatureSpace


class _ResNet18Binary(nn.Module):
    """torchvision ResNet-18 with a 2-way Sigmoid head (BCE); 3-channel input."""

    def __init__(self, pretrained=False):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Sequential(nn.Linear(512, 2), nn.Sigmoid())
        self.net = net

    def forward(self, x):
        # [B, 2] sigmoid probabilities (for BCE on one-hot targets)
        return self.net(x)

    def features(self, x):
        # 512-d penultimate vector = the input to fc (matches EvaluateCNN's hook)
        n = self.net
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = n.layer4(x)
        x = n.avgpool(x)
        return torch.flatten(x, 1)


class OODSupervisedCNN(Model):
    @staticmethod
    def _key():
        return "supervised-cnn"

    def init_model(self):
        pretrained = bool(self.options.get("pretrained", False))
        return _ResNet18Binary(pretrained=pretrained).to(self.device)

    def set_loss_function(self):
        return nn.BCELoss()

    @staticmethod
    def _to3(images):
        if isinstance(images, (list, tuple)):
            images = images[0]
        return torch.cat([images, images, images], dim=1)

    def _train_epoch(self, loader):
        self.model.train()
        losses = []
        for images, labels in tqdm(loader, disable=None):
            images = self._to3(images).to(self.device)
            targets = F.one_hot(labels.long().view(-1), num_classes=2).float().to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return sum(losses) / max(len(losses), 1)

    def _val_epoch(self, loader):
        self.model.eval()
        losses, ys, yhats = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, disable=None):
                images = self._to3(images).to(self.device)
                tgt = F.one_hot(labels.long().view(-1), num_classes=2).float().to(self.device)
                out = self.model(images)
                losses.append(self.loss_function(out, tgt).item())
                ys.extend(tgt.argmax(dim=1).cpu().tolist())
                yhats.extend(out[:, 1].cpu().tolist())
        auroc = roc_auc_score(ys, yhats) if len(set(ys)) > 1 else 0.0
        return sum(losses) / max(len(losses), 1), auroc

    def train_one_epoch(self, train_loader, val_loader):
        self.current_epoch_number += 1
        train_loss = self._train_epoch(train_loader)
        val_loss, val_auroc = self._val_epoch(val_loader)
        self.lr_scheduler.step()
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        if not hasattr(self, "best_auroc"):
            self.best_auroc = 0.0
        if val_auroc > self.best_auroc:
            self.best_auroc = val_auroc
            self.best_epoch_number = self.current_epoch_number
            self.save_model(epoch_val_loss=val_loss)
        if self.options.get("print_mode", True):
            print(f"[{self._key()}] Epoch {self.current_epoch_number}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  auroc={val_auroc:.4f}  best_auroc={self.best_auroc:.4f}")


class OODSupervisedCNNFeatureSpace(FeatureSpace):
    def get_features(self, dset):
        model = self.feature_model.model
        model.eval()
        device = self.feature_model.device
        feats = []
        loader = DataLoader(dset, batch_size=32, shuffle=False)
        with torch.no_grad():
            for images, _ in loader:
                if isinstance(images, (list, tuple)):
                    images = images[0]
                images = torch.cat([images, images, images], dim=1)   # [B, 3, 28, 28]
                images = images.to(device)
                z = model.features(images).detach().cpu().numpy()     # [B, 512] penultimate
                feats.append(z)
        return np.vstack(feats)


__all__ = ["OODSupervisedCNN", "OODSupervisedCNNFeatureSpace"]
