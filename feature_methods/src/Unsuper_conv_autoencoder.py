"""
Implementation of convolutional autoencoder architecture and model class.
Source: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""
import numpy as np  # For numerical computations
from tqdm import tqdm  # For progress bars
import torch  # PyTorch core library
import torch.nn as nn  # For defining neural network layers
from torch.utils.data import DataLoader, Dataset  # For data loading and handling

from .base import Model  # Base model class
from .base import FeatureSpace  # Base feature extraction class

# Define the Encoder architecture
class Encoder(nn.Module):
    """
    Encoder module for the convolutional autoencoder.
    Compresses input images into a latent representation.
    """
    def __init__(self, c_hid=16, latent_dim=100):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            # Convolutional layers with downsampling (stride=2)
            nn.Conv2d(3, c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output to a 1D vector
            nn.Linear(2 * 16 * c_hid, latent_dim),  # Fully connected layer for the latent space
        )
    
    def forward(self, x):
        """
        Forward pass of the encoder.
        - x: Input image tensor
        - Returns the latent representation.
        """
        return self.network(x)

# Define the Decoder architecture
class Decoder(nn.Module):
    """
    Decoder module for the convolutional autoencoder.
    Reconstructs images from the latent representation.
    """
    def __init__(self, c_hid=16, latent_dim=100):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),  # Map latent space back to 4x4x(2 * c_hid)
            nn.ReLU()
        )
        self.network = nn.Sequential(
            # Transpose convolutional layers for upsampling
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid, 3, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Tanh(),  # Tanh activation for normalized output (-1 to 1)
        )
    
    def forward(self, x):
        """
        Forward pass of the decoder.
        - x: Latent representation tensor
        - Returns the reconstructed image tensor.
        """
        z = self.linear(x)  # Map latent space to intermediate representation
        z = z.reshape(z.shape[0], -1, 4, 4)  # Reshape to match the convolutional layers
        return self.network(z)

# Combine Encoder and Decoder into the Autoencoder
class ConvAutoEncoderCore(nn.Module):
    """
    Convolutional Autoencoder core that combines the encoder and decoder.
    """
    def __init__(self, c_hid=16, latent_dim=100):
        super(ConvAutoEncoderCore, self).__init__()
        self.encoder = Encoder(c_hid, latent_dim)  # Encoder module
        self.decoder = Decoder(c_hid, latent_dim)  # Decoder module
    
    def forward(self, x):
        """
        Forward pass for the autoencoder.
        - Encodes the input and reconstructs it.
        """
        return self.decoder(self.encoder(x))
    
    def features(self, x: torch.Tensor):
        """
        Extracts latent features from the encoder.
        - x: Input tensor
        - Returns the latent feature representation.
        """
        return self.encoder(x)

# Autoencoder model class (inherits from Model)
class ConvAutoEncoder(Model):
    """
    Autoencoder model class for training and evaluation.
    """
    def init_model(self) -> ConvAutoEncoderCore:
        """
        Initializes the convolutional autoencoder model with given hyperparameters.
        """
        return ConvAutoEncoderCore(
            c_hid=self.options["c_hid"],  # Number of hidden channels
            latent_dim=self.options["latent_dim"],  # Dimensionality of latent space
        ).to(self.device)

    def set_loss_function(self):
        """
        Sets the loss function for training.
        - Mean Squared Error (MSE) is used to compare reconstructed and original images.
        """
        return torch.nn.MSELoss()
    
    def train_one_epoch(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Trains the autoencoder for one epoch and evaluates on validation data.
        - Updates weights using MSE loss.
        - Saves the best model based on validation loss.
        """
        # Update the current epoch counter
        self.current_epoch_number += 1

        # Training phase
        self.model.train()
        total_train_loss = []  # List to store training loss for each batch

        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # Clear optimizer gradients
            self.optimizer.zero_grad()

            # Convert grayscale images to 3-channel
            images = torch.cat([images, images, images], dim=1)
            images = images.to(self.device)

            # Forward pass: compute reconstructed images
            outputs = self.model(images)

            # Compute loss (MSE between input and reconstructed images)
            batch_loss = self.loss_function(images, outputs)

            # Backpropagation and optimizer step
            batch_loss.backward()
            self.optimizer.step()

            # Record training loss
            total_train_loss.append(batch_loss.item())
        
        # Calculate average training loss for the epoch
        epoch_train_loss = sum(total_train_loss) / len(total_train_loss)

        # Validation phase
        self.model.eval()
        total_val_loss = []  # List to store validation loss for each batch

        for i, (images, labels) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                # Convert grayscale images to 3-channel
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.device)

                # Forward pass: compute reconstructed images
                outputs = self.model(images)

                # Compute validation loss
                batch_loss = self.loss_function(images, outputs)
                total_val_loss.append(batch_loss.item())
        
        # Calculate average validation loss for the epoch
        epoch_val_loss = sum(total_val_loss) / len(total_val_loss)

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

        # Print training and validation loss
        if self.options["print_mode"]:
            print(
                f"Epoch #{self.current_epoch_number} | Train Loss: {epoch_train_loss:.4f} | " +
                f"Val Loss: {epoch_val_loss:.4f} | Best Loss: {self.best_loss:.4f}"
            )

    @staticmethod
    def _key():
        """
        Returns a unique key identifier for this model class.
        """
        return "conv-autoencoder"

# Feature extraction class (inherits from FeatureSpace)
class ConvAutoEncoderFeatureSpace(FeatureSpace):
    """
    Class for extracting latent features using a trained convolutional autoencoder.
    """
    def get_features(self, dset: Dataset):
        """
        Extracts feature embeddings for a dataset.
        - Uses the encoder part of the autoencoder.
        """
        self.feature_model.model.eval()  # Set model to evaluation mode
        features = []  # List to store extracted features
        ldr = DataLoader(dset, batch_size=32, shuffle=False)  # DataLoader for the dataset

        for j, (images, labels) in enumerate(tqdm(ldr)):
            with torch.no_grad():  # Disable gradient computation
                # Convert grayscale images to 3-channel
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.feature_model.device)

                # Extract latent features
                fts = self.feature_model.model.features(images).detach().cpu().numpy()
                features.append(fts)

        # Combine all features into a single NumPy array
        return np.vstack(features)
