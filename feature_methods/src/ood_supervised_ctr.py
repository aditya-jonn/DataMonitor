"""Supervised-contrastive feature extractor — faithful to the author's
train_contrastive.py (the contrastive-ood repo that produced the paper).

Self-contained: embeds the canonical SupContrast network (SupConResNet over a
CIFAR-style ResNet: 3x3 stride-1 stem, in_channel=3, 512-d encoder) and the
canonical multi-view SupConLoss (Khosla et al.; impl. Yonglong Tian). This
deliberately does NOT use the public repo's homemade SupContrastive_Feature
loss (a different, flat objective) nor a torchvision resnet18 (whose 7x7
stride-2 stem + maxpool over-downsamples a 28x28 image).

Recipe (matches the author's scripts):
  - SupConResNet(name=base_model, head=projection): encoder 512-d, mlp head 128-d,
    forward returns an L2-normalized projection
  - two AUGMENTED crops (RandomResizedCrop(28) + RandomRotation(10)) via
    TwoCropTransform, applied in datasets/__init__.py for method=="supervised-ctr";
    grayscale tripled to 3 channels
  - loss = SupConLoss(temp) on features shaped [B, 2, 128]
  - OOD features = L2-normalized 512-d ENCODER output (matches EvaluateCTR)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.base import Model, FeatureSpace


# ---------------------------------------------------------------------------
# Canonical SupContrast network (networks/resnet_big.py), reproduced inline.
# Source: https://github.com/HobbitLong/SupContrast
# ---------------------------------------------------------------------------
class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def _resnet18(**kwargs):
    return _ResNet(_BasicBlock, [2, 2, 2, 2], **kwargs)


_MODEL_DICT = {"resnet18": (_resnet18, 512)}


class _SupConResNet(nn.Module):
    """Backbone encoder + projection head; forward returns L2-normalized projection."""

    def __init__(self, name="resnet18", head="mlp", feat_dim=128):
        super().__init__()
        model_fun, dim_in = _MODEL_DICT[name]
        self.encoder = model_fun()
        if head == "linear":
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError("head not supported: {}".format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


# ---------------------------------------------------------------------------
# Canonical multi-view SupConLoss (requires features shaped [bsz, n_views, dim]).
# ---------------------------------------------------------------------------
class _SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")
        if len(features.shape) < 3:
            raise ValueError("features needs to be [bsz, n_views, ...], at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both labels and mask")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class OODSupervisedCTR(Model):
    @staticmethod
    def _key():
        return "supervised-ctr"

    def init_model(self):
        name = self.options.get("base_model") or "resnet18"
        head = self.options.get("projection") or "mlp"
        return _SupConResNet(name=name, head=head, feat_dim=128).to(self.device)

    def set_loss_function(self):
        return _SupConLoss(temperature=self.options.get("temp", 0.07))

    def _run_split(self, loader, train: bool):
        running, n = 0.0, 0
        for images, labels in tqdm(loader):
            # TwoCropTransform yields [view1, view2]; stack both views along batch.
            if isinstance(images, (list, tuple)):
                images = torch.cat([images[0], images[1]], dim=0)
            else:  # defensive; should not happen during ctr training
                images = torch.cat([images, images], dim=0)
            # grayscale -> 3 channels (author tripled the single channel)
            images = torch.cat([images, images, images], dim=1)
            images = images.to(self.device)
            labels = labels.to(self.device).long().view(-1)
            bsz = labels.shape[0]
            if train:
                self.optimizer.zero_grad()
            feats = self.model(images)                                  # [2B, 128] normalized
            f1, f2 = torch.split(feats, [bsz, bsz], dim=0)
            feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [B, 2, 128]
            loss = self.loss_function(feats, labels)
            if train:
                loss.backward()
                self.optimizer.step()
            running += loss.item() * bsz
            n += bsz
        return running / max(n, 1)

    def train_one_epoch(self, train_loader, val_loader):
        self.current_epoch_number += 1
        self.model.train()
        epoch_train = self._run_split(train_loader, train=True)
        self.model.eval()
        with torch.no_grad():
            epoch_val = self._run_split(val_loader, train=False)
        self.lr_scheduler.step()
        self.train_loss_history.append(epoch_train)
        self.val_loss_history.append(epoch_val)
        if epoch_val < self.best_loss:
            self.best_loss = epoch_val
            self.best_epoch_number = self.current_epoch_number
            self.save_model(epoch_val_loss=epoch_val)
        if self.options.get("print_mode", True):
            print(f"Epoch {self.current_epoch_number}  train={epoch_train:.4f}  "
                  f"val={epoch_val:.4f}  best={self.best_loss:.4f}")


class OODSupervisedCTRFeatureSpace(FeatureSpace):
    def get_features(self, dset):
        model = self.feature_model.model
        model.eval()
        device = self.feature_model.device
        feats = []
        loader = DataLoader(dset, batch_size=32, shuffle=False)
        with torch.no_grad():
            for images, _ in loader:
                # extraction uses method="none" (single images); stay robust to a list.
                if isinstance(images, (list, tuple)):
                    images = images[0]
                images = torch.cat([images, images, images], dim=1)   # [B, 3, 28, 28]
                images = images.to(device)
                z = model.encoder(images)                             # [B, 512]
                z = F.normalize(z)                                    # matches EvaluateCTR
                feats.append(z.detach().cpu().numpy())
        return np.vstack(feats)


__all__ = ["OODSupervisedCTR", "OODSupervisedCTRFeatureSpace"]
