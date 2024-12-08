"""
Package `__init__` file with `load_model` function
"""
from torch.utils.data import Dataset

from .base import Model
from .base import FeatureSpace
from .conv_autoencoder import ConvAutoEncoder
from .conv_autoencoder import ConvAutoEncoderFeatureSpace
from .ood_supervised_cnn import OODSupervisedCNN
from .ood_supervised_cnn import OODSupervisedCNNFeatureSpace
from .ood_supervised_ctr import OODSupervisedCTR
from .ood_supervised_ctr import OODSupervisedCTRFeatureSpace


def load_model(options: dict, mode="training") -> Model:
    if options["method"] == ConvAutoEncoder._key():
        model = ConvAutoEncoder(options)
    elif options["method"] == OODSupervisedCNN._key():
        model = OODSupervisedCNN(options)
    elif options["method"] == OODSupervisedCTR._key():
        model = OODSupervisedCTR(options)
    else:
        raise NotImplementedError(f'requested model not implemented: {options["method"]}')
    
    if mode == "testing":
        model.load(options["save_path"])

    return model


def load_eval(model: Model, training_set: Dataset, \
              validation_set: Dataset, testing_set: Dataset) -> FeatureSpace:
    if model.options["method"] == ConvAutoEncoder._key():
        evl = ConvAutoEncoderFeatureSpace(model, training_set, validation_set, testing_set)
    elif model.options["method"] == OODSupervisedCNN._key():
        evl = OODSupervisedCNNFeatureSpace(model, training_set, validation_set, testing_set)
    elif model.options["method"] == OODSupervisedCTR._key():
        evl = OODSupervisedCTRFeatureSpace(model, training_set, validation_set, testing_set)
    else:
        raise NotImplementedError(f"requested model's feature space not implemented: {model.options['method']}")

    return evl