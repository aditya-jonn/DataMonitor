# Modules Overview

The **DataMonitor** pipeline consists of three key modules designed to detect Out-of-Distribution (OOD) inputs and monitor data drift over time. Each module serves a specific purpose and incorporates advanced methodologies to address real-world challenges in medical imaging datasets.

---

## Module 1: Feature Extraction Module

The **Feature Extraction Module** transforms raw image data into structured feature representations that can be analyzed quantitatively. This module supports both **supervised** and **unsupervised** learning methods, accommodating datasets with or without labels. Additionally, it includes radiomics feature extraction for clinically relevant features.


### **Key Features**

1. **Supervised Learning**:
   - **Pretrained Models**:
     - **Method1**: Use general-purpose CNNs (e.g., VGG16, ResNet) as feature extractors. These pretrained networks project high-dimensional image data into a reduced, meaningful feature space. Features extracted from reference datasets are used to monitor shifts without the need for a task-specific model.
     - **Method 2**: Use task-specific models trained for a specific objective (e.g., pneumonia detection in chest X-rays). These models leverage learned representations that align closely with the specific task, enabling better detection of shifts related to the task.

   - **Contrastive Learning**:
     - A supervised contrastive learning approach is used to distinguish in-distribution (reference) data from out-of-distribution (test) samples.
     - The model is trained to:
       - Maximize similarity among samples from the same distribution.
       - Minimize similarity with out-of-distribution samples.
     - This method creates a specialized feature space optimized for robust OOD detection.

2. **Unsupervised Learning**:
   - Uses techniques such as:
     - Autoencoders to learn compressed representations of the data and identify deviations in reconstruction errors.
     - Dimensionality reduction (e.g., PCA, t-SNE, UMAP) to visualize and identify clusters or anomalies.



3. **Radiomics Features**:
   - Extracts domain-specific features from medical images, such as:
     - **Texture**: Gray-Level Co-occurrence Matrix (GLCM), Gray-Level Run Length Matrix (GLRLM).
     - **Shape**: Compactness, Sphericity, Ellipticity.
     - **Intensity**: Histogram-based features (mean, standard deviation, skewness).
     - **Spatial Patterns**: Edge density, spatial heterogeneity.

---

## Module 2: OOD Metric Generation 

The **OOD Metric Generation Module** quantifies deviations between reference (in-distribution) and test (potentially out-of-distribution) data. This module employs **similarity-based metrics** and **distance-based metrics** to measure shifts in data distributions effectively. These metrics can be computed over raw data, feature embeddings, or outputs from a trained model. Examples of the metrics in **DataMonitor** include:


#### 1. **Cosine Similarity**
- **Description**: Measures the cosine of the angle between two feature vectors, indicating their directional alignment.
- **Formula**:  
  ![Cosine Similarity](https://latex.codecogs.com/png.latex?%5Ctext%7BCosine%20Similarity%7D%20%3D%20%5Cfrac%7B%5Cvec%7BA%7D%20%5Ccdot%20%5Cvec%7BB%7D%7D%7B%5C%7C%5Cvec%7BA%7D%5C%7C%20%5Ccdot%20%5C%7C%5Cvec%7BB%7D%5C%7C%7D)  
  where \( \vec{A} \) and \( \vec{B} \) are the two feature vectors, and \( \|\vec{A}\| \) is the Euclidean norm of \( \vec{A} \).
- **Range**: [-1, 1]
  - \( 1 \): Identical vectors.
  - \( 0 \): Orthogonal vectors.
  - \( -1 \): Opposite vectors.


#### 2. **Correlation Coefficient**
- **Description**: Computes the linear correlation between two datasets. A high positive correlation indicates similar distributions.
- **Formula**:  
  ![Correlation Coefficient](https://latex.codecogs.com/png.latex?r%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28x_i%20-%20%5Cbar%7Bx%7D%29%28y_i%20-%20%5Cbar%7By%7D%29%7D%7B%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28x_i%20-%20%5Cbar%7Bx%7D%29%5E2%20%5Ccdot%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_i%20-%20%5Cbar%7By%7D%29%5E2%7D%7D)  
  where \( x \) and \( y \) are the two datasets, and \( \bar{x} \), \( \bar{y} \) are their means.
- **Range**: [-1, 1]
  - \( 1 \): Perfect positive correlation.
  - \( 0 \): No correlation.
  - \( -1 \): Perfect negative correlation.


#### 3. **Maximum Mean Discrepancy (MMD)**
- **Description**: Measures the difference between the mean embeddings of two distributions \( p \) and \( q \) in a reproducing kernel Hilbert space (RKHS).
- **Formula**:  
  ![MMD Formula](https://latex.codecogs.com/png.latex?%5Ctext%7BMMD%7D%5E2%20%3D%20%5Cfrac%7B1%7D%7Bm%5E2-m%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%5Csum_%7Bj%20%5Cneq%20i%7D%5E%7Bm%7D%20%5Ckappa%28z_i%2C%20z_j%29%20%2B%20%5Cfrac%7B1%7D%7Bn%5E2-n%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Csum_%7Bj%20%5Cneq%20i%7D%5E%7Bn%7D%20%5Ckappa%28z%27_i%2C%20z%27_j%29%20-%20%5Cfrac%7B2%7D%7Bmn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%20%5Ckappa%28z_i%2C%20z%27_j%29)  
  where \( \{z_i\}_{i=1}^m \sim p \), \( \{z'_i\}_{i=1}^n \sim q \), and \( \kappa \) is a kernel function.
- **Kernel Example (RBF)**:  
  ![RBF Kernel](https://latex.codecogs.com/png.latex?%5Ckappa%28z%2C%20%5Ctilde%7Bz%7D%29%20%3D%20%5Cexp%5Cleft%28-%5Cfrac%7B%7C%7Cz-%5Ctilde%7Bz%7D%7C%7C%5E2%7D%7B%5Csigma%7D%5Cright%29%29)  
  where \( \sigma \) is the median pairwise distance.
- **Range**: [0, ∞]
  - Lower values indicate more similarity.


#### 4. **Mahalanobis Distance**
- **Description**: Quantifies the distance between a point and a distribution, accounting for covariance.
- **Formula**:  
  ![Mahalanobis Distance](https://latex.codecogs.com/png.latex?d%5E2%20%3D%20%28x-%5Cmu%29%5ET%20%5CSigma%5E%7B-1%7D%20%28x-%5Cmu%29)  
  where \( \mu \) is the mean vector, and \( \Sigma \) is the covariance matrix.
- **Range**: [0, ∞]
  - Lower values indicate closer proximity to the reference distribution.


#### 5. **Earth Mover's Distance (EMD)**
- **Description**: Quantifies the minimum cost of transforming one distribution into another, interpreted as the "work" required to align two distributions.
- **Range**: [0, ∞]
  - Lower values indicate greater similarity.

### **Post-hoc and Model-Agnostic Analysis**

The **OOD Metric Generation Module** provides flexibility by enabling post-hoc, model-agnostic analysis. This means the metrics can be applied independently of the specific model or task. These 

- These metrics can quantify deviations between reference and test distributions without requiring task-specific models.
- They can help pinpoint out-of-distribution images in datasets or inputs during deployment.
- They can detect and flag anomalies in live data streams for manual review.
- Facilitate comparison between datasets, such as identifying differences in cross-modality (e.g., CT vs. X-ray) or cross-institutional data (e.g., datasets collected at different medical centers).

---

## Module 3: Statistical Process Control for Drift Monitoring 

The **SPC Module** monitors dataset stability over time and flags deviations using **Statistical Process Control (SPC)** techniques. It ensures that potential data drift or anomalies are promptly identified.

### **Key Methods and Thresholding**

#### **1. Sigma Rules**
- **Description**: Detects outliers by identifying data points that fall beyond a specified number of standard deviations (\( k \)) from the mean.
- **Threshold**:
  - Common defaults:
    - \( \pm 2\sigma \): Approximately 95% of data within bounds.
    - \( \pm 3\sigma \): Approximately 99.7% of data within bounds.
  - **Optimization**:
    - Thresholds can be optimized based on:
      - **False Positive Rate (FPR)**: A higher \( k \)-value reduces false positives but may miss subtle drifts.
      - **Application Sensitivity**: For critical applications (e.g., clinical safety), stricter thresholds (\( \pm 3\sigma \)) are preferred.
  - **User Input**:
    - Users can specify the \( k \)-value to tailor sensitivity.
    - Defaults to \( \pm 3\sigma \) in absence of user input.


#### **2. Cumulative Sum (CUSUM)**
- **Description**: Tracks cumulative deviations from a target value or baseline mean over time, making it effective for detecting small, persistent changes.
- **Formula**:
  ![CUSUM Formula](https://latex.codecogs.com/svg.latex?C_t%20=%20%5Cmax%280,%20C_%7Bt-1%7D%20+%20%28x_t%20-%20K%29%29)
  where:
  - \( C_t \): Cumulative sum at time \( t \),
  - \( x_t \): Observed value at time \( t \),
  - \( K \): Reference or drift allowance.
- **Threshold**:
  - Trigger is raised when \( C_t \) exceeds a pre-specified limit \( h \).
  - Default values:
    - \( K = 0.5\sigma \) (small allowable drift).
    - \( h = 5 \sigma \) (detection threshold).
  - **Optimization**:
    - Select \( K \) and \( h \) to balance:
      - Early detection of drifts.
      - Minimizing false alarms.
    - Simulations or historical data can be used to fine-tune thresholds for specific datasets.
  - **User Input**:
    - Users can override \( K \) and \( h \) values.
    - Default settings are designed for moderate sensitivity.


#### **3. Bernoulli CUSUM**
- **Description**: Monitors binary outcomes (e.g., OOD detected or not) in real-time and flags when the observed rate significantly deviates from the expected baseline.
- **Formula**:
  ![Bernoulli CUSUM Formula](https://latex.codecogs.com/svg.latex?C_t%20=%20%5Cmax%280,%20C_%7Bt-1%7D%20+%20%5Clog%5Cleft%28%5Cfrac%7Bp_1%7D%7Bp_0%7D%5Cright%29%20-%20%5Clog%5Cleft%28%5Cfrac%7B1-p_1%7D%7B1-p_0%7D%5Cright%29%29)
  where:
  - \( p_0 \): Baseline probability,
  - \( p_1 \): Drift probability,
  - \( C_t \): Cumulative score at time \( t \).
- **Threshold**:
  - Trigger is raised when \( C_t \) exceeds a pre-specified limit \( h \).
  - Default values:
    - \( p_0 = 0.01 \): Assumes 1% baseline OOD rate.
    - \( p_1 = 0.05 \): Detects when OOD rate rises to 5%.
    - \( h = 4 \): Moderate sensitivity to changes.
  - **Optimization**:
    - Fine-tune \( p_0 \), \( p_1 \), and \( h \) using simulations to reflect real-world data.
    - Adjust \( p_1 \) higher for systems tolerant to OODs and lower for sensitive applications.
  - **User Input**:
    - Users can input \( p_0 \), \( p_1 \), and \( h \) based on application needs.
    - Defaults ensure robust detection in general settings.


### **Threshold Selection Guidelines**
1. **Default Settings**: Pre-configured thresholds (\( \pm 3\sigma \), \( K = 0.5\sigma \), \( h = 4 \)) ensure balanced sensitivity and robustness for most applications.
2. **User-Defined Inputs**:
   - Users with domain expertise can customize thresholds to meet specific operational or regulatory needs.
3. **Optimization**:
   - Utilize historical data or simulations to adjust thresholds.
   - Trade-offs:
     - **Sensitivity vs. Specificity**: Stricter thresholds detect subtle drifts but increase false positives.
     - **Detection Delay vs. False Alarms**: Tighter thresholds reduce detection delays but may increase noise.

---

