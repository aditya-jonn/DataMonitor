"""
Implementations of model + feature extraction for OOD supervised contrastive learning (SupCon).
"""
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .base import Model
from .base import FeatureSpace
from .supcon_loss import SupConLoss
from .resnet_big import SupConResNet


class OODSupervisedCTR(Model):
    def init_model(self):
        # create model
        model = SupConResNet(
            name=self.options["base_model"],
            head=self.options["projection"]
        )
        # GPU support
        if self.device == "cuda":
            # no parallelization
            model = model.cuda()
            cudnn.benchmark = True
        return model
    
    def set_loss_function(self):
        loss_fxn = SupConLoss(temperature=self.options["temp"])
        if torch.cuda.is_available():
            return loss_fxn.cuda()
        else:
            return loss_fxn
        
    def train_one_epoch(self, train_loader: DataLoader, val_loader: DataLoader):
        # update epoch number
        self.current_epoch_number += 1
        # training phase
        self.model.train()
        total_train_loss = []
        for j, (images, labels) in enumerate(tqdm(train_loader)):
            # clear gradients
            self.optimizer.zero_grad()
            # stack both images (loader uses TwoCropTransform)
            images = torch.cat([images[0], images[1]], dim=0)
            # make grayscale into 3 channel
            images = torch.cat([images, images, images], dim=1)
            # forward pass and partition output features
            images = images.to(self.device)
            labels = labels.to(self.device)
            features = self.model(images)
            bsz = labels.shape[0]
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # supervised contrastive (SupCon) loss
            batch_loss = self.loss_function(features, labels)
            # update parameters
            batch_loss.backward()
            self.optimizer.step()
            # store per-batch loss
            total_train_loss.append(batch_loss.item())
        # compute epoch train loss
        epoch_train_loss = sum(total_train_loss) / len(total_train_loss)
        # validation phase
        self.model.eval()
        total_val_loss = []
        for j, (images, labels) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                # stack both images
                images = torch.cat([images[0], images[1]], dim=0)
                # make grayscale into 3 channel
                images = torch.cat([images, images, images], dim=1)
                # forward pass
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = self.model(images)
                bsz = labels.shape[0]
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                batch_loss = self.loss_function(features, labels)
                # store per-batch loss
                total_val_loss.append(batch_loss.item())
        # compute epoch val loss
        epoch_val_loss = sum(total_val_loss) / len(total_val_loss)
        # update lr
        self.lr_scheduler.step()
        # update model
        if self.best_loss > epoch_val_loss:
            self.best_loss = epoch_val_loss
            self.best_epoch_number = self.current_epoch_number
            self.save_model(
                epoch_val_loss=epoch_val_loss,
            )
        # save loss history
        self.train_loss_history.append(epoch_train_loss)
        self.val_loss_history.append(epoch_val_loss)
        # print
        if self.options["print_mode"]:
            print(
                f"Epoch #{self.current_epoch_number} | Train Loss: {epoch_train_loss:.4f} | " + 
                f"Val Loss: {epoch_val_loss:.4f} | Best Loss: {self.best_loss:.4f} | "
            )

    @staticmethod
    def _key():
        return "supervised-ctr"


class OODSupervisedCTRFeatureSpace(FeatureSpace):
    def get_features(self, dset):
        self.feature_model.model.eval()
        features = []
        ldr = DataLoader(dset, batch_size=32, shuffle=False)
        for j, (images, labels) in enumerate(tqdm(ldr)):
            with torch.no_grad():
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.feature_model.device)
                outputs = self.feature_model.model.encoder(images)
                norm_outputs = F.normalize(outputs)
                features.append(norm_outputs.detach().cpu().numpy())
        return np.vstack(features)
