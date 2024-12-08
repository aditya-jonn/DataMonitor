"""
Implementation of OOD (Out-of-Distribution) supervised CNN feature extraction.
"""
from copy import deepcopy
from tqdm import tqdm  # For progress bars
import numpy as np  # For numerical computations
import torch  # PyTorch core library
from torch.utils.data import DataLoader  # For loading datasets
from torchvision.models import resnet18, ResNet18_Weights  # Pretrained ResNet-18 model
from sklearn.metrics import roc_auc_score  # For AUROC calculation

from .base import Model  # Base model class
from .base import FeatureSpace  # Base feature extraction class

class OODSupervisedCNN(Model):
    """
    Class implementing a supervised CNN for OOD detection.
    Uses a modified ResNet-18 architecture with feature extraction and supervised training.
    """

    # Define a forward hook to capture intermediate features
    def get_inputs(self, name):
        """
        Hook function to extract intermediate layer outputs from ResNet-18.
        - `name`: Name to associate with the hooked layer's output.
        """
        def hook(model, inpt, output):
            # Save the input to the hooked layer (features before logits)
            self.cnn_layers[name] = inpt[0].detach()
        return hook

    def init_model(self):
        """
        Initializes the CNN model based on ResNet-18.
        - Loads pretrained weights if specified in options.
        - Modifies the fully connected layer for binary classification.
        """
        # Load ResNet-18 architecture
        model = resnet18(
            weights=(ResNet18_Weights.DEFAULT if self.options["pretrained"] else None)
        )
        # Replace the fully connected layer with a binary classification head
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=2),  # Two output classes
            torch.nn.Sigmoid()  # Sigmoid activation for probabilities
        )
        # Attach a forward hook to the final logits layer for feature extraction
        self.cnn_layers = {}  # Dictionary to store layer outputs
        self.hook = model.fc.register_forward_hook(self.get_inputs('fts'))  # Hook for feature extraction

        # Move model to the appropriate device (GPU/CPU)
        model = model.to(self.device)
        return model
    
    def set_loss_function(self):
        """
        Sets the loss function for training.
        - Uses Binary Cross Entropy Loss (BCELoss) for binary classification.
        """
        return torch.nn.BCELoss()
    
    def train_one_epoch(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Trains the CNN for one epoch and evaluates on the validation set.
        - Computes loss, AUROC, and accuracy for validation.
        - Saves the best model based on validation loss.
        """
        # Update epoch counter
        self.current_epoch_number += 1

        # Training phase
        self.model.train()
        total_train_loss = []  # Store training loss per batch

        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # Clear gradients
            self.optimizer.zero_grad()

            # Convert grayscale images to 3-channel format
            images = torch.cat([images, images, images], dim=1)
            images = images.to(self.device)

            # Convert labels to one-hot encoding for binary classification
            targets = torch.nn.functional.one_hot(labels, num_classes=2).float()
            targets = targets.to(self.device)

            # Forward pass: compute outputs
            outputs = self.model(images)

            # Compute loss (BCELoss)
            batch_loss = self.loss_function(outputs, targets)

            # Backward pass and optimizer step
            batch_loss.backward()
            self.optimizer.step()

            # Record training loss for this batch
            total_train_loss.append(batch_loss.item())

        # Calculate average training loss for the epoch
        epoch_train_loss = sum(total_train_loss) / len(total_train_loss)

        # Validation phase
        self.model.eval()
        total_val_loss = []  # Store validation loss per batch
        val_ys = []  # True labels
        val_yhats = []  # Predicted probabilities for class 1

        for i, (images, labels) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                # Convert grayscale images to 3-channel format
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.device)

                # Convert labels to one-hot encoding
                targets = torch.nn.functional.one_hot(labels, num_classes=2).float()
                targets = targets.to(self.device)

                # Forward pass: compute outputs
                outputs = self.model(images)

                # Compute validation loss
                batch_loss = self.loss_function(outputs, targets)
                total_val_loss.append(batch_loss.item())

                # Collect true labels and predicted probabilities for AUROC calculation
                val_ys.extend(targets.detach().cpu().argmax(dim=1).tolist())
                val_yhats.extend(outputs[:, 1].detach().squeeze().cpu().tolist())

        # Calculate average validation loss for the epoch
        epoch_val_loss = sum(total_val_loss) / len(total_val_loss)

        # Calculate AUROC for validation
        epoch_val_auroc = roc_auc_score(val_ys, val_yhats)

        # Calculate validation accuracy
        val_ys = np.array(val_ys)
        val_preds = np.array(val_yhats) > 0.5  # Threshold probability > 0.5
        epoch_val_acc = np.sum(val_ys == val_preds) / val_ys.shape[0]

        # Update learning rate
        self.lr_scheduler.step()

        # Save the best model based on validation loss
        if self.best_loss > epoch_val_loss:
            self.best_loss = epoch_val_loss
            self.best_epoch_number = self.current_epoch_number
            self.save_model(epoch_val_loss=epoch_val_loss)

        # Save loss history
        self.train_loss_history.append(epoch_train_loss)
        self.val_loss_history.append(epoch_val_loss)

        # Print epoch metrics
        if self.options["print_mode"]:
            print(
                f"Epoch #{self.current_epoch_number} | Train Loss: {epoch_train_loss:.4f} | " +
                f"Val Loss: {epoch_val_loss:.4f} | Best Loss: {self.best_loss:.4f} | " +
                f"Val AUROC: {epoch_val_auroc:.4f} | Val Acc: {epoch_val_acc:.4f}"
            )

    @staticmethod
    def _key():
        """
        Returns a unique identifier for this model class.
        """
        return "supervised-cnn"

class OODSupervisedCNNFeatureSpace(FeatureSpace):
    """
    Class for extracting feature embeddings using the supervised CNN model.
    """
    def get_features(self, dset):
        """
        Extracts features from the CNN's intermediate layer for a given dataset.
        """
        self.feature_model.model.eval()  # Set the model to evaluation mode
        features = []  # Store extracted features
        ldr = DataLoader(dset, batch_size=32, shuffle=False)  # Create a DataLoader for the dataset

        for j, (images, labels) in enumerate(tqdm(ldr)):
            with torch.no_grad():
                # Convert grayscale images to 3-channel format
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.feature_model.device)

                # Forward pass through the model to populate cnn_layers
                outputs = self.feature_model.model(images)

                # Extract features from the hooked intermediate layer
                features.append(self.feature_model.cnn_layers['fts'].detach().cpu().numpy())

        # Combine features into a single NumPy array
        return np.vstack(features)
