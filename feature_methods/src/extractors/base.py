"""
Implementations of base abstract `Model` and `FeatureSpace` class.
"""
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class Model(ABC):
    def __init__(self, options: dict):
        self.options = options
        # set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # set model (and move to device)
        self.model = self.init_model()
        # loss function
        self.loss_function = self.set_loss_function()
        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.options["learning_rate"])
        # lr scheduler
        milestones = [0.5 * self.options["max_epochs"], 0.75 * self.options["max_epochs"]]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        # tracking variables
        self.best_loss = np.inf
        self.best_epoch_number = -1
        self.current_epoch_number = 0
        self.train_loss_history = []
        self.val_loss_history = []

    @abstractmethod
    def init_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def set_loss_function(self):
        pass

    def load(self, model_path: str):
        print("==> Loading!")
        d = torch.load(model_path)
        # set options
        self.options = d["options"]
        # load model & device
        self.model = self.init_model()
        try:
            self.model.load_state_dict(d["model"])
        except Exception:
            raise RuntimeError("Incorrect model preconditions provided.")
        # load optimizer
        self.optimizer.load_state_dict(d["optimizer"])
        # load last epoch & loss
        self.current_epoch_number = d["epoch"]
        self.best_epoch_number = d["epoch"]
        self.best_loss = d["val_loss"]

    @abstractmethod
    def train_one_epoch(self, train_loader: DataLoader, val_loader: DataLoader):
        pass

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(1, self.options["max_epochs"]+1):
            self.train_one_epoch(train_loader, val_loader)

    # source: https://github.com/HobbitLong/SupContrast/blob/master/util.py
    def save_model(self, epoch_val_loss):
        print("==> Saving!")
        state = {
            "method": self._key(),
            "model": deepcopy(self.model).state_dict(),
            "model_name": os.path.basename(self.options["save_path"]),
            "device": self.device,
            "options": self.options,
            "optimizer": deepcopy(self.optimizer).state_dict(),
            "epoch": self.current_epoch_number,
            "val_loss": epoch_val_loss
        }
        torch.save(state, self.options["save_path"])
        del state

    def name(self):
        return os.path.basename(self.options["save_path"])
    
    @staticmethod
    @abstractmethod
    def _key(self):
        pass


class FeatureSpace(ABC):
    def __init__(self, model: Model,
                 training_set: Dataset,
                 validation_set: Dataset,
                 testing_set: Dataset):
        # store feature extraction model
        self.feature_model = model

        # calculate features
        self.train_features = self.get_features(training_set)
        self.val_features = self.get_features(validation_set)
        self.test_features = self.get_features(testing_set)
        
    @abstractmethod
    def get_features(self, dset: Dataset):
        pass
    
    def save_features(self, save_path: str):
        print(f"==> Saving features at {save_path} !")
        state = {
            "method": self.model._key(),
            "model_path": self.model.options["save_path"],
            "model_name": self.model.name(),
            "train_features": self.train_features,
            "val_features": self.val_features,
            "test_features": self.test_features,
        }
        torch.save(state, save_path)
        del state
