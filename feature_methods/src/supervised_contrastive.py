"""
Implementations of model and feature extraction for OOD supervised contrastive learning (SupCon).
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

# Supervised contrastive learning class for OOD detection
class OODSupervisedCTR(Model):
    # Initialize the model
    def init_model(self):
        # Create a supervised contrastive learning model using ResNet backbone
        model = SupConResNet(
            name=self.options["base_model"],  # ResNet model type (e.g., ResNet50)
            head=self.options["projection"]  # Type of projection head (e.g., MLP)
        )
        # If GPU is available, enable CUDA support and optimization
        if self.device == "cuda":
            model = model.cuda()
            cudnn.benchmark = True  # Optimize for consistent input size
        return model
    
    # Set the supervised contrastive loss function
    def set_loss_function(self):
        loss_fxn = SupConLoss(temperature=self.options["temp"])  # SupCon loss with a temperature parameter
        if torch.cuda.is_available():
            return loss_fxn.cuda()
        else:
            return loss_fxn
        
    # Train the model for one epoch
    def train_one_epoch(self, train_loader: DataLoader, val_loader: DataLoader):
        self.current_epoch_number += 1  # Increment epoch count
        self.model.train()  # Set model to training mode
        total_train_loss = []  # Initialize list to store training losses

        # Training phase
        for j, (images, labels) in enumerate(tqdm(train_loader)):
            self.optimizer.zero_grad()  # Clear gradients
            # Stack two views of the images (from TwoCropTransform)
            images = torch.cat([images[0], images[1]], dim=0)
            # Convert grayscale images to 3-channel format
            images = torch.cat([images, images, images], dim=1)
            images = images.to(self.device)  # Move images to the device (GPU/CPU)
            labels = labels.to(self.device)  # Move labels to the device

            # Forward pass through the model and partition the output features
            features = self.model(images)
            bsz = labels.shape[0]  # Batch size
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)  # Split features into two views
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # Combine views for contrastive loss

            # Compute supervised contrastive loss
            batch_loss = self.loss_function(features, labels)
            batch_loss.backward()  # Backpropagation
            self.optimizer.step()  # Update model parameters
            total_train_loss.append(batch_loss.item())  # Store batch loss

        # Compute average training loss for the epoch
        epoch_train_loss = sum(total_train_loss) / len(total_train_loss)

        # Validation phase
        self.model.eval()  # Set model to evaluation mode
        total_val_loss = []  # Initialize list to store validation losses
        for j, (images, labels) in enumerate(tqdm(val_loader)):
            with torch.no_grad():  # No gradient computation for validation
                images = torch.cat([images[0], images[1]], dim=0)  # Stack two views
                images = torch.cat([images, images, images], dim=1)  # Convert grayscale to 3-channel
                images = images.to(self.device)  # Move images to the device
                labels = labels.to(self.device)  # Move labels to the device
                features = self.model(images)  # Forward pass
                bsz = labels.shape[0]  # Batch size
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)  # Split features
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # Combine views
                batch_loss = self.loss_function(features, labels)  # Compute loss
                total_val_loss.append(batch_loss.item())  # Store batch loss

        # Compute average validation loss for the epoch
        epoch_val_loss = sum(total_val_loss) / len(total_val_loss)

        # Adjust the learning rate
        self.lr_scheduler.step()

        # Save the model if validation loss improves
        if self.best_loss > epoch_val_loss:
            self.best_loss = epoch_val_loss
            self.best_epoch_number = self.current_epoch_number
            self.save_model(epoch_val_loss=epoch_val_loss)

        # Save training and validation loss history
        self.train_loss_history.append(epoch_train_loss)
        self.val_loss_history.append(epoch_val_loss)

        # Print loss information if enabled
        if self.options["print_mode"]:
            print(
                f"Epoch #{self.current_epoch_number} | Train Loss: {epoch_train_loss:.4f} | " + 
                f"Val Loss: {epoch_val_loss:.4f} | Best Loss: {self.best_loss:.4f} | "
            )

    @staticmethod
    def _key():
        return "supervised-ctr"


# Feature space extraction class for OOD supervised contrastive learning
class OODSupervisedCTRFeatureSpace(FeatureSpace):
    # Extract features from a dataset
    def get_features(self, dset):
        self.feature_model.model.eval()  # Set model to evaluation mode
        features = []  # Initialize list to store features
        ldr = DataLoader(dset, batch_size=32, shuffle=False)  # DataLoader for the dataset

        for j, (images, labels) in enumerate(tqdm(ldr)):
            with torch.no_grad():  # No gradient computation during feature extraction
                images = torch.cat([images, images, images], dim=1)  # Convert grayscale to 3-channel
                images = images.to(self.feature_model.device)  # Move images to the device
                outputs = self.feature_model.model.encoder(images)  # Pass images through the encoder
                norm_outputs = F.normalize(outputs)  # Normalize features
                features.append(norm_outputs.detach().cpu().numpy())  # Move features to CPU and detach
        return np.vstack(features)  # Combine all features into a single array
