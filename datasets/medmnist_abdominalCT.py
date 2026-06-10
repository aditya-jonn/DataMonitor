"""
Implementation of MedMNIST AbnominalCT dataset class.
Source: https://medmnist.com/ (see Organ{A/C/S}MNIST)
"""
import json
import random

import medmnist
from medmnist import INFO
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

class AbnominalCTDataset(Dataset):
    def __init__(self, data_dir, split, label_mode, positive_dataset="organamnist", \
                 tfms=None, random_seed=1001):
        # local RNG for the index shuffle; do NOT touch global RNG state
        rng = random.Random(random_seed)

        # set dataset transform
        self.tfms = tfms

        self.data_flags = [
            "organamnist", "organcmnist", "organsmnist"
        ]
        self.data_order = {
            "organamnist":0, "organcmnist":1, "organsmnist":2
        }
        self.data_list = [
            getattr(medmnist, INFO[self.data_flags[i]]['python_class'])\
                (split=split, transform=self.tfms, download=True, root=data_dir, as_rgb=False)\
                for i in range(len(self.data_flags))
        ]
        # all three datasets have the same task & label
        self.label_dict = INFO[self.data_flags[0]]['label']
        self.n_classes = len(self.label_dict)

        self.l1, self.l2, self.l3 = [len(x) for x in self.data_list]
        self.length = self.l1 + self.l2 + self.l3
        self.random_order = rng.sample(range(self.length), k=self.length)

        if label_mode not in ["unsupervised", "cheap-supervised", "full-supervised"]:
            raise ValueError("label mode not supported: {}".format(label_mode))
        self.label_mode = label_mode
        if self.label_mode == "cheap-supervised":
            if positive_dataset not in self.data_flags:
                raise ValueError("dataset not supported: {}".format(positive_dataset))
            self.positive = self.data_order[positive_dataset]
            self.positive_dataset = positive_dataset

    def __getitem__(self, index):
        ri, ai = self.reduce_index(self.random_order[index])
        image, class_label = self.data_list[ai][ri]
        if self.label_mode == "unsupervised":
            return image
        elif self.label_mode == "cheap-supervised":
            if ai == self.positive:
                dataset_label = 1
            else:
                dataset_label = 0
            return image, dataset_label
        else:
            return image, class_label, self.data_flags[ai] 

    # reduce 'global' index to the correct 'internal dataset' index
    def reduce_index(self, j):
        if j < self.l1:
            return j, 0
        elif (j < self.l1 + self.l2):
            return j - self.l1, 1
        elif (j < self.length):
            return j - self.l1 - self.l2, 2
        else:
            raise ValueError("index is out of range: {}".format(j))

    def __len__(self):
        return self.length
    
    @staticmethod
    def get_default_transform() -> torchvision.transforms.Compose:
        dataset_transforms = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])
        ])
        return dataset_transforms
