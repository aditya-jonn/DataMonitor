#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:36:17 2024

@author: ghada

This module computes various metrics for analyzing the similarity or dissimilarity
between two sets of features, typically representing different distributions.
It supports calculations of cosine, mahalanobis, euclidean, geodesic distances,
and more complex metrics like Jensen-Shannon divergence, Kullback-Leibler divergence,
Earth Mover's Distance, and Maximum Mean Discrepancy.
"""

from scipy.spatial import distance
from scipy.stats import entropy
from scipy.linalg import pinv
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.spatial.distance import jensenshannon
import ot  # Optimal Transport

def compute_cosine_similarity(tr_features, tt_features):
    centroid = np.mean(tr_features, axis=0)
    return [1 - distance.cosine(feature, centroid) for feature in tt_features]

def compute_mahalanobis_similarity(tr_features, tt_features):
    covariance_matrix = np.cov(tr_features, rowvar=False)
    covariance_matrix_inv = pinv(covariance_matrix)
    centroid = np.mean(tr_features, axis=0)
    return [distance.mahalanobis(feature, centroid, covariance_matrix_inv) for feature in tt_features]

def compute_euclidean_similarity(tr_features, tt_features):
    centroid = np.mean(tr_features, axis=0)
    return [-distance.euclidean(feature, centroid) for feature in tt_features]

def compute_jensen_shannon(tr_prob_dist, tt_prob_dist):
    return jensenshannon(tr_prob_dist, tt_prob_dist)

def compute_kullback_leibler(tr_prob_dist, tt_prob_dist):
    return entropy(tr_prob_dist, tt_prob_dist)

def compute_earth_movers(tr_features, tt_features):
    # Assuming that features are normalized histograms or similar distributions
    return ot.emd2(tr_features, tt_features, metric='euclidean')

def compute_maximum_mean_discrepancy(tr_features, tt_features, kernel='rbf'):
    # Placeholder function for MMD, assuming an RBF kernel by default
    # Implementation of MMD would require a full function to calculate this metric
    pass

# Add other functions as needed...

# Example usage within the module if run as a script
if __name__ == "__main__":
    tr_features = np.random.rand(10, 5)
    tt_features = np.random.rand(8, 5)
    print("Cosine Similarity:", compute_cosine_similarity(tr_features, tt_features))
    print("Jensen-Shannon Divergence:", compute_jensen_shannon(np.random.rand(10,), np.random.rand(10,)))
