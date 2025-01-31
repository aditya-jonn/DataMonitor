# Distribution Similarity Metrics Module

This Python module is designed to compute various metrics for analyzing the similarity or dissimilarity between two sets of features, typically representing different distributions. The extracted features from your datasets will be used to compute all the included metrics. Below is a detailed explanation of each function and its application:

## Available Functions

- **compute_similarity**: This function calculates similarity scores between training and testing feature sets based on a specified similarity type. It supports 'cosine', 'mahalanobis', 'euclidean', and 'geodesic' distances.
- **compute_kolmogorov_smirnov**: Applies the Kolmogorov-Smirnov test to determine if two samples come from the same distribution, focusing on the maximum distance between their empirical distribution functions.
- **compute_jensen_shannon**: Calculates the Jensen-Shannon divergence between two probability distributions, providing a symmetric and bounded measure of similarity.
- **compute_kullback_leibler**: Computes the Kullback-Leibler divergence to measure how one probability distribution diverges from a second, reference probability distribution.
- **compute_bhattacharyya**: Measures the overlap between two statistical samples or distributions, useful for pattern recognition and classification tasks.
- **compute_earth_movers**: Calculates the Earth Mover's Distance (Wasserstein distance) which measures the minimum cost of transforming one distribution into another.
- **compute_entropy**: Determines the entropy of predictions to quantify uncertainty and detect out-of-distribution data.
- **compute_maximum_mean_discrepancy**: Computes the Maximum Mean Discrepancy (MMD) which measures the distance between the mean embeddings of two distributions in a high-dimensional space.

## Usage Example

To use these functions, import the module and pass the feature arrays for your training and testing sets. Here is an example of how to use the `compute_similarity` function:

```python
import numpy as np
from your_module import compute_similarity

# Example feature arrays
training_features = np.random.rand(10, 5)  # 10 samples, 5 features each
testing_features = np.random.rand(8, 5)    # 8 samples, 5 features each

# Compute cosine similarity
results = compute_similarity(training_features, testing_features, 'cosine')
print(results)
