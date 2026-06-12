"""
Training script
"""
import argparse
import os
import json
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_data
from feature_methods import load_model
from utils import make_run_dir

with open("cfg.json", "r") as f:
    cfg = json.load(f)

def parse_options():
    parser = argparse.ArgumentParser('command-line arguments for model training')

    # general arguments
    parser.add_argument("--use-gpus", default='all', type=str, help='gpu device numbers')
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DM_SEED", "1001")),
                        help="random seed for reproducible training")
    # method arguments
    parser.add_argument('--method', type=str, \
                        choices=["conv-autoencoder", "supervised-cnn", "supervised-ctr"])
    parser.add_argument('--c_hid', type=int, default=16, help='number of hidden channels')
    parser.add_argument('--latent_dim', type=int, default=100, help='number of latent dims')
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained weights (for supervised CNN)')
    parser.add_argument('--base_model', type=str,\
                        choices=["resnet18"], help="base CNN architecture (for CNN-based methods)")
    parser.add_argument('--projection', type=str, default='mlp',
                        choices=['linear', 'mlp'], help='projection head for CLR')
    parser.add_argument('--temp', type=float, default=0.07, help="temperature for CLR loss fxn")


    # data arguments
    parser.add_argument('--dataset', choices=["MedMNIST-AbdominalCT"])
    parser.add_argument('--positive_dataset', type=str, default='organamnist',
                            help='which dataset is in-distribution')

    # training loop arguments
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--print_mode', type=bool, default=True)

    # extract
    opt = parser.parse_args()

    # set GPU visibility
    if opt.use_gpus != 'all':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.use_gpus

    # get in-distribution CT view
    views = {
        "A": "Axial",
        "C": "Coronal",
        "S": "Sagittal"
    }
    id_view = views[opt.positive_dataset.replace("organ", "")[0].capitalize()]

    # storage files
    opt.model_path = './model_saves'
    run_dir = make_run_dir(opt.model_path, opt.batch_size, opt.seed)
    opt.model_name = '{}_{}_{}_defaulttfms_lr{}_bsz{}_nep{}_indist{}_time{}'.\
        format(opt.method, opt.base_model, opt.dataset, opt.learning_rate, opt.batch_size, opt.max_epochs, id_view, time.time())
    opt.save_path = os.path.join(run_dir, opt.model_name + ".pt")

    # make options dictionary
    options = {
        "data_dir": cfg["data_dir"],
        "method": opt.method,
        "c_hid": opt.c_hid,
        "latent_dim": opt.latent_dim,
        "pretrained": opt.pretrained,
        "base_model": opt.base_model,
        "projection": opt.projection,
        "temp": opt.temp,
        "dataset": opt.dataset,
        "positive_dataset": opt.positive_dataset,
        "batch_size": opt.batch_size,
        "max_epochs": opt.max_epochs,
        "learning_rate": opt.learning_rate,
        "print_mode": opt.print_mode,
        "save_path": opt.save_path,
        "model_name": opt.model_name,
        "seed": opt.seed,
    }

    return options

def _seed_everything(seed=None):
    import os, random
    import numpy as np
    import torch
    if seed is None:
        seed = int(os.environ.get('DM_SEED', '1001'))
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('[seed] deterministic mode, seed=' + str(seed))


def _seed_worker(worker_id):
    import numpy as np, random, torch
    s = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(s)
    random.seed(s)


def main():
    options = parse_options()
    _seed_everything(options["seed"])

    train_set, val_set, test_set = load_data(options)

    g = torch.Generator().manual_seed(options["seed"])
    nw = int(os.environ.get("DM_NUM_WORKERS", "0"))   # 0 = old behavior if unset
    loader_kwargs = dict(num_workers=nw, persistent_workers=(nw > 0),
                         pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker)
    train_loader = DataLoader(train_set, batch_size=options["batch_size"], shuffle=True, generator=g, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=options["batch_size"], shuffle=True, generator=g, **loader_kwargs)

    feature_model = load_model(options)
    feature_model.train_model(train_loader, val_loader)

    # done
    print(f'==> Model name: {options["model_name"]} !')
    print(f'==> Model is saved at: {options["save_path"]} !')

if __name__ == "__main__":
    main()
