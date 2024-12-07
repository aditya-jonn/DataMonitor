## Feature Representation Methods for OOD and Drift Monitoring

The first step in out-of-distribution (OOD) detection and dataset drift monitoring is converting images into a suitable feature space. We have implemented three methods for feature representation.

After extracting the features, various methods for measuring differences in the feature space can be applied to detect OOD and drift. 

---

## Feature Representation Methods

### 1. Feature-Based Detection Using Pretrained Models
This method uses pretrained or task-specific models to extract meaningful feature representations from images.

#### Scenario 1:
- A **pretrained CNN** (e.g., VGG16, ResNet) is used as a feature extractor to project high-dimensional image data into a reduced, meaningful feature space.
- The features are extracted from the reference dataset, allowing for general-purpose dataset drift monitoring.
- This scenario does not require a task-specific model and is suitable for detecting data shifts across diverse applications.

#### Scenario 2:
- Features are extracted using a **task-specific model** trained for a particular objective (e.g., distinguishing pneumonia from normal cases in chest X-rays).
- This approach leverages the task model's learned representations, making it better suited for detecting shifts specific to the task at hand.

---

### 2. Supervised Contrastive Learning
- A contrastive learning approach is used to train a model to distinguish **in-distribution** (reference) from **out-of-distribution** (test) samples.
- The model is trained to:
  - **Maximize similarity** between samples from the same distribution.
  - **Minimize similarity** with samples from different distributions.
- This method creates a highly specialized feature space for robust OOD detection.

---

### 3. Radiomics Feature Extraction
- **Traditional radiomics features** are extracted to quantify key attributes of the images, such as texture, intensity, shape, and spatial patterns.
- Radiomics features are particularly valuable in medical imaging because:
  - They provide domain-specific insights aligned with clinical relevance.
  - They offer interpretability, complementing deep learning-based features.

---

## How to Use This Repository

This repository contains three Jupyter notebooks, each implementing one of the feature representation methods. Each notebook extracts features from the input dataset and saves them for further analysis:

1. **Notebook for Feature-Based Detection Using Pretrained Models**:
   - Demonstrates both general-purpose and task-specific feature extraction.
   - Saves the extracted features in a specified folder.

2. **Notebook for Supervised Contrastive Learning**:
   - Explains and implements a contrastive learning framework for creating a specialized feature space.
   - Saves the extracted features in a designated folder.

3. **Notebook for Radiomics Feature Extraction**:
   - Walks through the process of extracting radiomics features from medical images.
   - Saves the extracted features in a designated folder.


---

## Next Steps

Once features are extracted, methods for measuring differences in the feature space can be applied to detect dataset drift.
