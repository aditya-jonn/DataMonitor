# Distribution Similarity Metrics Module

This Python module is designed to compute various metrics for analyzing the similarity or dissimilarity between two sets of features, typically representing different distributions. Users can choose to compute one or multiple metrics to monitor and analyze the differences between a reference distribution and other data distributions. Below is a detailed explanation of each function and its application, along with the strengths of each type of metric:

## Available Functions

- **compute_similarity**: Calculates similarity scores between reference and test feature sets based on a specified similarity type. Supports 'cosine', 'mahalanobis', 'euclidean', and 'geodesic' distances.
  - **Cosine Similarity**: Measures the cosine of the angle between two vectors, ideal for text analysis where orientation rather than magnitude matters.
  - **Mahalanobis Distance**: Considers the covariance among the variables to identify the similarity, making it suitable for detecting outliers when features are correlated.
  - **Euclidean Distance**: Computes the root of square differences, effective for naturally clustered data.
  - **Geodesic Distance**: Measures the shortest path between points on any surface, useful for geographic data.

- **compute_kolmogorov_smirnov**: Applies the Kolmogorov-Smirnov test to determine if two samples come from the same distribution. It is powerful in detecting differences in both the location and shape of the empirical cumulative distribution functions of two samples.
  
- **compute_jensen_shannon**: Calculates the Jensen-Shannon divergence, providing a symmetric and bounded measure of similarity. This method is especially good for comparing probability distributions as it is smoother and bounded between 0 and 1.
  
- **compute_kullback_leibler**: Computes the Kullback-Leibler divergence, which measures how one probability distribution diverges from a second, reference probability distribution. It is particularly useful for measuring the amount of information lost when using one distribution to approximate another.
  
- **compute_bhattacharyya**: Measures the overlap between two statistical samples or distributions. It is effective in applications involving classification as it quantifies the separability between the distributions.
  
- **compute_earth_movers**: Calculates the Earth Mover's Distance, which is a form of Optimal Transport. It measures the minimal cost of transforming one distribution into another, making it extremely useful for comparing images or multidimensional data.
  
- **compute_entropy**: Determines the entropy of predictions to quantify uncertainty and detect out-of-distribution data. This metric is particularly useful when dealing with probabilistic models.
  
- **compute_maximum_mean_discrepancy**: Computes the Maximum Mean Discrepancy (MMD) which measures the distance between the mean embeddings of two distributions in a high-dimensional space. This is useful for testing whether two samples are drawn from the same distribution.

- **compute_optimal_transport**: Calculates the Optimal Transport distance between two distributions, which can be thought of as the minimum cost required to transform one distribution into another. It's particularly powerful in domains where the geometric structure of the data (e.g., images) is important.

## Usage Example

To use these functions, import the module and pass the feature arrays for your reference and test sets. Here is an example of how to use the `compute_similarity` function:

```python
import numpy as np
from your_module import compute_similarity

# Example feature arrays
reference_features = np.random.rand(10, 5)  # 10 samples, 5 features each
test_features = np.random.rand(8, 5)        # 8 samples, 5 features each

# Compute cosine similarity
results = compute_similarity(reference_features, test_features, 'cosine')
print(results)
