"""OOD Detection using features, code adapted from Ghada"""
import argparse
import json
import os
import random
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from scipy.linalg import pinv
from tqdm import tqdm

from utils import set_seed

with open("cfg.json", "r") as f:
    cfg = json.load(f)

# fancy metric name
metrics = {
    "cosine": "Cosine Similarity",
    "mahalanobis": "Mahalanobis Distance"
}

"""Apply SPC rules (per image)"""
def apply_spc_rules(data, mean, std, metric_name):
    """Detects SPC rule violations."""
    violations = {"Rule 1": []}

    for i in range(len(data)):
        # Rule 1: Any single data point more than 3σ in absolute distance from the center line.
        # For cosine similarity: only check the lower bound
        # For Mahalanobis distance: only check the upper bound 
        if metric_name == "cosine":
            if data[i] < (mean - 3 * std):
                violations["Rule 1"].append(i)
        elif metric_name == "mahalanobis":
            if data[i] > (mean + 3 * std):
                violations["Rule 1"].append(i)
        else:
            raise NotImplementedError(f"requested metric does not ahve Rule 1 implemented: {metric_name}")

    return violations

"""Core OOD visualization function"""
def ood_visualization(distances, mean, UCL, LCL, rule, ood_labels=None, metric_name=None, figure_path="./figs/viz.png"):
    def make_confusion_matrix(vls, ood_true):
        ood_pred = [0 for _ in range(len(ood_true))]
        for idx in vls:
            ood_pred[idx] = 1
        return confusion_matrix(ood_true, ood_pred)

    plt.figure(figsize=(16, 6))
    plt.plot(distances, color='black', marker='o', markersize=4, linestyle='-')
    plt.axhline(y=mean, color='black', linestyle='-')
    if metric_name == "cosine":
        plt.axhline(y=np.clip(UCL, a_min=0.0, a_max=1.0), color='black', linestyle='--')
        plt.axhline(y=np.clip(LCL, a_min=0.0, a_max=1.0), color='black', linestyle='--')
        plt.fill_between(range(len(distances)), \
                        np.clip(LCL, a_min=0.0, a_max=1.0), \
                        np.clip(UCL, a_min=0.0, a_max=1.0), \
                        color='grey', alpha=0.1)
    else:
        plt.axhline(y=UCL, color='black', linestyle='--')
        plt.axhline(y=LCL, color='black', linestyle='--')
        plt.fill_between(range(len(distances)), \
                        LCL, \
                        UCL, \
                        color='grey', alpha=0.1)

    # For presentation purposes, disable axes
    # plt.xlabel('Image Sequence', fontsize=12)
    # plt.ylabel(metrics[metric_name], fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    
    plt.xticks([])
    plt.yticks([])

    # Assuming you have a function apply_spc_rules() which was not provided in your code
    violations = apply_spc_rules(distances, mean, (UCL - mean) / 3, metric_name)

    for idx in violations[rule]:
        plt.plot(idx, distances[idx], '*', color='grey', markersize=16, label='Auto OOD')

    if ood_labels is not None:
        for i, is_out in enumerate(ood_labels):
            if is_out:
                plt.plot(i, distances[i], marker='o', markersize=16, linestyle='None', color='black', mfc='none', label='Actual OOD')
        """Compute confusion matirx"""
        cfm = make_confusion_matrix(violations[rule], ood_labels)
        tn, fp, fn, tp = cfm.ravel()
        # Specificity
        if tn + fp == 0:
            specificity = np.nan
        else:
            specificity = tn / (tn + fp)
        # Sensitivity
        if tp + fn == 0:
            sensitivity = np.nan
        else:
            sensitivity = tp / (tp + fn)
        # Acc
        acc = (tp + tn) / (tp + tn + fp + fn)
        # Print
        print(f"Accuracy: {acc:.4f} | Specificity: {specificity:.4f} | Sensitivity: {sensitivity:.4f}")

    plt.xlim(0, len(distances) - 1)
    if metric_name == "cosine":
        plt.ylim(0, 1.5)

    # Ensure no duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.savefig(figure_path)

# Compute cosine similarity
def compute_cosine_similarity(tr_features_, tt_features_):
    centroid = np.mean(tr_features_, axis=0)
    similarities = []
    epsilon = np.random.normal(loc=1e-6, scale=1e-3, size=centroid.shape[0])
    for feature in tt_features_:
        # Add a small epsilon to make sure cosine distances are not undefined
        similarities.append(
            # Clip the cosine distance to be in (0,1)
            1 - np.clip(distance.cosine(feature + epsilon, centroid), a_min=0.0, a_max=1.0)
        )
    return similarities

# Compute mahalanobis distance
def compute_mahalanobis_distance(tr_features_, tt_features_):
    centroid = np.mean(tr_features_, axis=0)
    cov_matrix = np.cov(tr_features_, rowvar=False)
    inv_cov_matrix = pinv(cov_matrix)

    distances = []
    for feature in tt_features_:
        diff = feature - centroid
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
        distances.append(dist)
    return distances

"""Compute control limits"""
def compute_control_limits(tr_features, tt_features, metric):
    # Control similarities
    if metric == "cosine":
        train_distances = compute_cosine_similarity(tr_features, tr_features)
        test_distances = compute_cosine_similarity(tr_features, tt_features)
    elif metric == "mahalanobis":
        train_distances = compute_mahalanobis_distance(tr_features, tr_features)
        test_distances = compute_mahalanobis_distance(tr_features, tt_features)
    else:
        raise NotImplementedError(f"Metric is not implemented: {metric}")
    # Control limit calculations
    train_mean = np.mean(train_distances)
    train_std = np.std(train_distances)
    train_UCL = train_mean + 3 * train_std
    train_LCL = train_mean - 3 * train_std

    # Return
    return train_distances, test_distances, train_mean, train_std, train_UCL, train_LCL

"""OOD statistics"""
def ood_statistics(tr_features, tt_features, ood_labels, metric, n=100, rule="Rule 1"):
    # Precompute training similarities
    if metric == "cosine":
        fxn = compute_cosine_similarity
    elif metric == "mahalanobis":
        fxn = compute_mahalanobis_distance
    else:
        raise NotImplementedError(f"Metric is not implemented: {metric}")
    train_distances = fxn(tr_features, tr_features)
    # Control limit calculations
    train_mean = np.mean(train_distances)
    train_std = np.std(train_distances)
    # Bootstrap arrays
    accuracy = []
    sensitivity = []
    specificity = []
    # Bootstrap loop
    for i in tqdm(range(n)):
        # Pick a subset of the testing images
        sample = np.random.randint(low=0, high=tt_features.shape[0], size=500)
        tt_subset = tt_features[sample, :]
        ood_labels_subset = ood_labels[sample]
        # Calculate test similarities on subset
        tt_subset_distances = fxn(tr_features, tt_subset)
        # Calculate OOD detection accuracy/precision/sensitivity/specificity
        violations = apply_spc_rules(tt_subset_distances, train_mean, train_std, metric)
        ood_preds_subset = [0 for _ in range(len(tt_subset_distances))]
        for jj in violations[rule]:
            ood_preds_subset[jj] = 1
        cmatrix = confusion_matrix(ood_labels_subset, ood_preds_subset)
        tn, fp, fn, tp = cmatrix.ravel()
        # Specificity
        try:
            specificity.append(tn / (tn + fp))
        except RuntimeWarning:
            specificity.append(np.nan)
        # Sensitivity
        try:
            sensitivity.append(tp / (tp + fn))
        except RuntimeWarning:
            sensitivity.append(np.nan)
        # Accuracy
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
    # Report bootstrap results
    print(
        f"Accuracy: {np.mean(accuracy):.4f} [{(np.mean(accuracy) - np.std(accuracy)):.4f}, {(np.mean(accuracy) + np.std(accuracy)):.4f}]"
    )
    print(
        f"Specificity: {np.nanmean(specificity):.4f} [{(np.nanmean(specificity) - np.nanstd(specificity)):.4f}, {(np.nanmean(specificity) + np.nanstd(specificity)):.4f}]"
    )
    print(
        f"Sensitivity: {np.nanmean(sensitivity):.4f} [{(np.nanmean(sensitivity) - np.nanstd(sensitivity)):.4f}, {(np.nanmean(sensitivity) + np.nanstd(sensitivity)):.4f}]"
    )
    # Return
    return accuracy, specificity, sensitivity

"""Parse args"""
def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--metric", type=str, choices=["cosine", "mahalanobis"], help="OOD metric")
    parser.add_argument("--method", type=str, choices=["autoencoder", "cnn", "ctr"], help="method to get features per image")
    parser.add_argument("--bootstrap", type=int, default=100, help="number of bootstrapped samples")

    # extract
    opt = parser.parse_args()

    # Make options dictionary
    options = {
        "metric": opt.metric,
        "method": opt.method,
        "positive_dataset": "organamnist", # hard coded
        "bootstrap": opt.bootstrap,
        "data_dir": cfg["data_dir"]
    }

    return options

"""Main"""
def main():
    options = parse_options()

    # Load data from npz
    data_splits_path = "./numpy_files/data_splits.npz"
    D = np.load(data_splits_path)
    Xtr = D["Xtr"]
    ytr = D["ytr"]
    Xvl = D["Xvl"]
    yvl = D["yvl"]
    Xtt = D["Xtt"]
    ytt = D["ytt"]
    print(f"Loaded data splits from: {data_splits_path} !")

    # Get features
    if options["method"] == "autoencoder":
        F = np.load('./numpy_files/autoencoder_features.npz')
        Ftr = F["autoencoder_Ftr"]
        Ftt = F["autoencoder_Ftt"]
    elif options["method"] == "cnn":
        F = np.load('./numpy_files/cnn_features.npz')
        Ftr = F["cnn_Ftr"]
        Ftt = F["cnn_Ftt"]
    elif options["method"] == "ctr":
        F = np.load('./numpy_files/ctr_features.npz')
        Ftr = F["ctr_Ftr"]
        Ftt = F["ctr_Ftt"]
    else:
        raise NotImplementedError(f"Requested feature method is not implemented: {options['method']}")
    
    # Get features for all in-distribution (ID) training images
    Ftr_in = Ftr[ytr == 1]
    # Compute OOD statistics using entire test set
    n = options["bootstrap"]
    accuracy, specificity, sensitivity = ood_statistics(Ftr_in, Ftt, 1 - ytt, options["metric"], n=n)
    np.savez(
        file=f"./numpy_files/{options['metric']}_{options['method']}_bootstrap.npz",
        accuracy=accuracy,
        specificity=specificity,
        sensitivity=sensitivity,
    )

    # Add to results table
    table_entry = {
        "Metric": options["metric"],
        "Method": options["method"],
        # Accuracy
        "Mean Accuracy": np.nanmean(accuracy),
        "LCL Accuracy": np.nanmean(accuracy) - np.nanstd(accuracy),
        "UCL Accuracy": np.nanmean(accuracy) + np.nanstd(accuracy),
        # Specificity
        "Mean Specificity": np.nanmean(specificity),
        "LCL Specificity": np.nanmean(specificity) - np.nanstd(specificity),
        "UCL Specificity": np.nanmean(specificity) + np.nanstd(specificity),
        # Sensitivity
        "Mean Sensitivity": np.nanmean(sensitivity),
        "LCL Sensitivity": np.nanmean(sensitivity) - np.nanstd(sensitivity),
        "UCL Sensitivity": np.nanmean(sensitivity) + np.nanstd(sensitivity),
    }
    if os.path.exists(cfg['table_path']):
        df = pd.read_csv(cfg['table_path'])
        table_entry_df = pd.DataFrame([table_entry])
        df = pd.concat([df, table_entry_df], ignore_index=True)
    else:
        df = pd.DataFrame([table_entry])
    df.to_csv(cfg['table_path'], header=True, index=False)

if __name__ == "__main__":
    set_seed(2001)
    main()