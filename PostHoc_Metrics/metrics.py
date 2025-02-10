import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy, wasserstein_distance, ks_2samp
from scipy.linalg import pinv
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
import ot  # Optimal Transport library

# ===========================
# 📌 Probability Distribution Conversion
# ===========================

def convert_to_probability_distribution(feature_vector):
    """
    Converts a feature vector into a probability distribution.

    🔹 Purpose:
        - Converts raw numerical feature vectors into valid probability distributions.
        - Required for metrics such as Kullback-Leibler and Jensen-Shannon Divergence.

    🔹 Computation:
        - Normalizes the input vector so that all values sum to 1.
        - Clips values to prevent division by zero.

    🔹 Args:
        feature_vector (np.ndarray): A 1D array representing feature values.

    🔹 Returns:
        np.ndarray: A normalized probability distribution (sum = 1).
    """
    feature_vector = np.clip(feature_vector, a_min=1e-10, a_max=None)  # Avoid log(0)
    return feature_vector / np.sum(feature_vector)  # Normalize

# ===========================
# 📌 Metric Computation Functions
# ===========================

def compute_cosine_similarity(tr_features, tt_features):
    """
    Computes cosine similarity between test features and the centroid of training features.

    🔹 Purpose:
        - Measures how similar two vectors are in terms of their direction.
        - Commonly used for comparing high-dimensional representations (e.g., embeddings).

    🔹 Computation:
        - Cosine similarity = 1 - cosine distance.
        - Similarity ranges from [0, 1] where 1 means identical direction.

    🔹 Args:
        tr_features (np.ndarray): Training feature vectors.
        tt_features (np.ndarray): Test feature vectors.

    🔹 Returns:
        list: Cosine similarity values for each test feature.
    """
    centroid = np.mean(tr_features, axis=0)
    return [1 - distance.cosine(feature, centroid) for feature in tt_features]


def compute_euclidean_similarity(tr_features, tt_features):
    """
    Computes the Euclidean distance between test features and the centroid of training features.

    🔹 Purpose:
        - Measures how far apart two vectors are in Euclidean space.
        - Useful for detecting significant differences in feature distributions.

    🔹 Computation:
        - Euclidean distance formula: sqrt(sum((x_i - y_i)^2))

    🔹 Args:
        tr_features (np.ndarray): Training feature vectors.
        tt_features (np.ndarray): Test feature vectors.

    🔹 Returns:
        list: Euclidean distance values for each test feature.
    """
    centroid = np.mean(tr_features, axis=0)
    return [distance.euclidean(feature, centroid) for feature in tt_features]


def compute_mahalanobis_similarity(tr_features, tt_features):
    """
    Computes the Mahalanobis distance between test features and the centroid of training features.

    🔹 Purpose:
        - Measures distance while accounting for feature correlations.
        - Useful for detecting outliers in high-dimensional spaces.

    🔹 Computation:
        - Mahalanobis distance formula:
          sqrt((x - μ)^T Σ^(-1) (x - μ))
        - Uses inverse covariance matrix for normalization.

    🔹 Args:
        tr_features (np.ndarray): Training feature vectors.
        tt_features (np.ndarray): Test feature vectors.

    🔹 Returns:
        list: Mahalanobis distance values for each test feature.
    """
    centroid = np.mean(tr_features, axis=0)
    cov_matrix = np.cov(tr_features, rowvar=False) + np.eye(tr_features.shape[1]) * 1e-6
    cov_inv = pinv(cov_matrix)
    return [distance.mahalanobis(feature, centroid, cov_inv) for feature in tt_features]


def compute_jensen_shannon(tr_features, tt_features):
    """
    Computes Jensen-Shannon divergence between probability distributions.

    🔹 Purpose:
        - Measures similarity between two probability distributions.
        - A symmetric variant of KL divergence.

    🔹 Computation:
        - JS divergence formula:
          0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P + Q) / 2.

    🔹 Args:
        tr_features (np.ndarray): Training probability distributions.
        tt_features (np.ndarray): Test probability distributions.

    🔹 Returns:
        list: Jensen-Shannon divergence values.
    """
    centroid = convert_to_probability_distribution(np.mean(tr_features, axis=0))
    return [distance.jensenshannon(convert_to_probability_distribution(feature), centroid) for feature in tt_features]


def compute_kullback_leibler(tr_features, tt_features):
    """
    Computes Kullback-Leibler (KL) divergence between probability distributions.

    🔹 Purpose:
        - Measures how much one probability distribution differs from another.
        - KL divergence is asymmetric.

    🔹 Computation:
        - KL(P || Q) = sum(P * log(P / Q))

    🔹 Args:
        tr_features (np.ndarray): Training probability distributions.
        tt_features (np.ndarray): Test probability distributions.

    🔹 Returns:
        list: KL divergence values.
    """
    centroid = convert_to_probability_distribution(np.mean(tr_features, axis=0))
    return [entropy(convert_to_probability_distribution(feature), centroid) for feature in tt_features]


def compute_earth_movers(tr_features, tt_features):
    """
    Computes Earth Mover’s Distance (Wasserstein Distance).

    🔹 Purpose:
        - Measures the minimum cost of transforming one distribution into another.
        - Used in probability distribution comparisons.

    🔹 Computation:
        - Uses Wasserstein distance.

    🔹 Args:
        tr_features (np.ndarray): Training feature vectors.
        tt_features (np.ndarray): Test feature vectors.

    🔹 Returns:
        list: Earth Mover’s Distance values.
    """
    centroid = np.mean(tr_features, axis=0)
    return [wasserstein_distance(feature, centroid) for feature in tt_features]


def compute_maximum_mean_discrepancy(tr_features, tt_features):
    """
    Computes Maximum Mean Discrepancy (MMD) using an RBF kernel.

    🔹 Purpose:
        - Measures the difference between two probability distributions in a high-dimensional space.
        - Used for domain adaptation.

    🔹 Computation:
        - MMD(X, Y) = mean(K(X, X)) + mean(K(Y, Y)) - 2 * mean(K(X, Y))

    🔹 Args:
        tr_features (np.ndarray): Training feature vectors.
        tt_features (np.ndarray): Test feature vectors.

    🔹 Returns:
        float: MMD value.
    """
    X_kernel = rbf_kernel(tr_features, tr_features)
    Y_kernel = rbf_kernel(tt_features, tt_features)
    XY_kernel = rbf_kernel(tr_features, tt_features)
    return float(np.mean(X_kernel) + np.mean(Y_kernel) - 2 * np.mean(XY_kernel))


def compute_entropy(tr_features):
    """
    Computes entropy of a probability distribution.

    🔹 Purpose:
        - Quantifies the amount of uncertainty or randomness in a distribution.

    🔹 Computation:
        - Shannon entropy formula: H(X) = -sum(P * log(P))

    🔹 Args:
        tr_features (np.ndarray): Feature vectors.

    🔹 Returns:
        list: Entropy values for each feature vector.
    """
    return [entropy(convert_to_probability_distribution(feature)) for feature in tr_features]


def compute_optimal_transport(P, Q, cost_matrix):
    """
    Computes Optimal Transport distance between two probability distributions.

    🔹 Purpose:
        - Measures how much mass must be moved to make two distributions identical.

    🔹 Computation:
        - Uses the Earth Mover’s Distance (EMD) with a cost matrix.

    🔹 Args:
        P (np.ndarray): Source probability distribution.
        Q (np.ndarray): Target probability distribution.
        cost_matrix (np.ndarray): Transport cost matrix.

    🔹 Returns:
        float: Optimal Transport distance.
    """
    P = np.array(P) / np.sum(P)
    Q = np.array(Q) / np.sum(Q)
    return float(ot.emd2(P, Q, cost_matrix))


def compute_bhattacharyya(tr_features, tt_features):
    """
    Computes the Bhattacharyya distance between two sets of feature distributions.

    Purpose:
        - Measures the similarity between two probability distributions.
        - Commonly used in classification tasks to compare the overlap between distributions.

    Computation:
        - Bhattacharyya distance formula:
            -log(sum(sqrt(P * Q))) 
          where P and Q are normalized probability distributions.

    Args:
        tr_features (np.ndarray): In-distribution feature vectors (N_samples, D_dim).
        tt_features (np.ndarray): Test feature vectors (M_samples, D_dim).

    Returns:
        list: Bhattacharyya distances for each test feature compared to the training centroid.
    """
    # Compute the centroid of the training features
    centroid = np.mean(tr_features, axis=0)

    # Convert the centroid into a normalized probability distribution
    centroid_prob = convert_to_probability_distribution(centroid)

    # Compute Bhattacharyya distance for each test feature
    distances = [
        -np.log(
            np.sum(
                np.sqrt(
                    convert_to_probability_distribution(feature) * centroid_prob
                )
            ) + 1e-10  # Add a small value to prevent log(0)
        )
        for feature in tt_features
    ]
    return distances


def compute_bhattacharyya(tr_features, tt_features):
    """Compute Bhattacharyya distance between probability distributions."""
    centroid = np.mean(tr_features, axis=0)
    centroid_prob = centroid / np.sum(centroid)  # Normalize

    return [
        -np.log(np.sum(np.sqrt((sample / np.sum(sample)) * centroid_prob)) + 1e-10)
        for sample in tt_features
    ]

def compute_entropy(tr_features):
    """Compute entropy of a probability distribution."""
    return [entropy(convert_to_probability_distribution(feature)) for feature in tr_features]

def compute_kolmogorov_smirnov(tr_features, tt_features):
    """Compute Kolmogorov-Smirnov test statistic."""
    centroid = convert_to_probability_distribution(np.mean(tr_features, axis=0))
    return [ks_2samp(convert_to_probability_distribution(feature), centroid).statistic for feature in tt_features]

#def compute_optimal_transport(tr_features, tt_features):
#    """Compute Optimal Transport distance using POT (Python Optimal Transport)."""
#    centroid = convert_to_probability_distribution(np.mean(tr_features, axis=0))
#    cost_matrix = np.abs(np.subtract.outer(centroid, centroid))  # Simple cost matrix
#    return [ot.emd2(convert_to_probability_distribution(feature), centroid, cost_matrix) for feature in tt_features]


def compute_optimal_transport(P, Q, cost_matrix):
    """Compute Optimal Transport distance between two probability distributions."""
    
    # Normalize P and Q so they sum to the same value
    P = np.array(P) / np.sum(P)
    Q = np.array(Q) / np.sum(Q)
    
    return float(ot.emd2(P, Q, cost_matrix))


def compute_share_of_drifted_components(in_features, ood_features, p_threshold=0.05):
    """
    Compute the share of embedding components that show statistically significant drift.
    
    Args:
        in_features (np.ndarray): In-distribution feature vectors (N_samples, D_dim).
        ood_features (np.ndarray): Out-of-distribution feature vectors (M_samples, D_dim).
        p_threshold (float): Significance level for detecting drift (default = 0.05).

    Returns:
        float: Proportion of embedding components that exhibit drift (range: 0 to 1).
    """
    num_components = in_features.shape[1]  # Number of embedding dimensions
    drifted_components = 0  # Counter for drifted components

    # Check for drift in each embedding dimension
    for i in range(num_components):
        p_value = ks_2samp(in_features[:, i], ood_features[:, i]).pvalue  # Kolmogorov-Smirnov test
        if p_value < p_threshold:  # If p-value is below threshold, consider it drifted
            drifted_components += 1

    return drifted_components / num_components  # Normalize by total dimensions
