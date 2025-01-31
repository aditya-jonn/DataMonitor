#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:36:17 2024

@author: ghada
"""

from scipy.spatial import distance
import numpy as np
from scipy.linalg import pinv
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarities between training and testing features based on the specified type
def compute_similarity(tr_features, tt_features, similarity_type='cosine'):
    """
    Parameters:
    - tr_features (array): Training feature vectors.
    - tt_features (array): Testing feature vectors.
    - similarity_type (str): Type of similarity to compute ('cosine', 'mahalanobis', 'euclidean', or 'geodesic').

    Returns:
    - dict: Contains computed similarities and basic statistics.
    """
    def compute_cosine_similarity(tr_features_, tt_features_):
        centroid = np.mean(tr_features_, axis=0)
        return [1 - distance.cosine(feature, centroid) for feature in tt_features_]

    def compute_mahalanobis_similarity(tr_features_, tt_features_):
        covariance_matrix = np.cov(tr_features_, rowvar=False)
        covariance_matrix_inv = pinv(covariance_matrix)
        centroid = np.mean(tr_features_, axis=0)
        return [distance.mahalanobis(feature, centroid, covariance_matrix_inv) for feature in tt_features_]

    def compute_euclidean_similarity(tr_features_, tt_features_):
        centroid = np.mean(tr_features_, axis=0)
        return [-distance.euclidean(feature, centroid) for feature in tt_features_]

    def compute_geodesic_similarity(tr_features_, tt_features_):
        centroid = np.mean(tr_features_, axis=0)
        return [-distance.geodesic(feature, centroid) for feature in tt_features_]

    # Compute similarities
    if similarity_type == 'cosine':
        similarities = compute_cosine_similarity(tr_features, tt_features)
    elif similarity_type == 'mahalanobis':
        similarities = compute_mahalanobis_similarity(tr_features, tt_features)
    elif similarity_type == 'euclidean':
        similarities = compute_euclidean_similarity(tr_features, tt_features)
    elif similarity_type == 'geodesic':
        similarities = compute_geodesic_similarity(tr_features, tt_features)
    else:
        raise ValueError("Invalid similarity type. Choose 'cosine', 'mahalanobis', 'euclidean', or 'geodesic'.")

    # Compute statistics
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    median_similarity = np.median(similarities)
    mad_similarity = np.median(np.abs(similarities - median_similarity))
    percentile_95 = np.percentile(similarities, 95)
    percentile_99 = np.percentile(similarities, 99)
    range_similarity = np.ptp(similarities)
    iqr_similarity = np.percentile(similarities, 75) - np.percentile(similarities, 25)

    return {
        'similarities': similarities,
        'mean': mean_similarity,
        'std': std_similarity,
        'median': median_similarity,
        'mad': mad_similarity,
        'percentile_95': percentile_95,
        'percentile_99': percentile_99,
        'range': range_similarity,
        'iqr': iqr_similarity
    }
