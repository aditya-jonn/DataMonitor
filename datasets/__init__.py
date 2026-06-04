"""
Package `__init__`: dataset loading function
"""
import numpy as np
from tqdm import tqdm

from torchvision import transforms
from .medmnist_abdominalCT import AbnominalCTDataset
from feature_methods.supcon_loss import TwoCropTransform

# Master function for loading datasets
def load_data(options: dict):
    if options["dataset"] == "MedMNIST-AbdominalCT":
        train_tfms = AbnominalCTDataset.get_default_transform()
        val_tfms = AbnominalCTDataset.get_default_transform()
        test_tfms = AbnominalCTDataset.get_default_transform()
        if options["method"] == "supervised-ctr":
            _ctr_aug = transforms.Compose([
                transforms.RandomResizedCrop(size=28),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5]),
            ])
            train_tfms = TwoCropTransform(_ctr_aug)
            val_tfms = TwoCropTransform(_ctr_aug)
        # train/val/test datasets
        train_set = AbnominalCTDataset(
            data_dir=options["data_dir"],
            label_mode="cheap-supervised",
            positive_dataset=options["positive_dataset"],
            tfms=train_tfms,
            split="train"
        )
        val_set = AbnominalCTDataset(
            data_dir=options["data_dir"],
            label_mode="cheap-supervised",
            positive_dataset=options["positive_dataset"],
            tfms=val_tfms,
            split="val"
        )
        test_set = AbnominalCTDataset(
            data_dir=options["data_dir"],
            label_mode="cheap-supervised",
            positive_dataset=options["positive_dataset"],
            tfms=test_tfms,
            split="test"
        )
        return train_set, val_set, test_set
    else:
        raise NotImplementedError(f"requested dataset is not available: {options['dataset']}")

# reformat dataset into X,y matrix/vector pair
def matrixify(dset, label_mode="cheap-supervised"):
    if label_mode!="cheap-supervised":
        raise NotImplementedError("label mode not implemented: {}".format(label_mode))
    X = np.empty((len(dset), 28*28))
    y = np.empty((len(dset)))
    for i in tqdm(range(len(dset))):
        image, label = dset[i]
        X[i,:] = image.flatten()
        y[i] = int(label)
    y = y.astype(np.int32)
    return X,y
