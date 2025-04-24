# Notebook Examples

The **DataMonitor** tool provides Jupyter notebooks to demonstrate its functionality in various scenarios. These examples cover different use cases for detecting Out-of-Distribution (OOD) images and monitoring data drift. Each notebook corresponds to a specific scenario and includes step-by-step instructions, input data, and outputs.

---

## **Available Scenarios**

1. **Within-Modality OOD (Axial vs. Non-Axial CT Images)**  
   - **Objective**: Differentiate between axial and non-axial CT images within the same modality.  
   - **Use Case**: Monitor data consistency and identify variations in CT imaging views.

2. **Within-Modality OOD (Image Quality)**  
   - **Objective**: Detect lower-quality images compared to high-quality reference images.  
   - **Use Case**: Ensure imaging quality remains consistent over time.

3. **Cross-Modality OOD**  
   - **Objective**: Identify differences between datasets from distinct imaging modalities, such as chest X-rays (CXR) vs. other radiographic imaging.  
   - **Use Case**: Detect cross modality data drift.

4. **Demographic OOD**  
   - **Objective**: Distinguish adult chest X-rays from pediatric chest X-rays.  
   - **Use Case**: Detect shifts in data distributions caused by demographic differences.

5. **Cross-Dataset OOD**  
   - **Objective**: Differentiate datasets collected from various institutions, even when they share the same modality (e.g., adult CXR datasets).  
   - **Use Case**: Identify institution-specific imaging differences.

---

## **Notebook Details**

### **Notebook Structure**  
Each notebook includes:  
- **Input Data**: The specific dataset used for the scenario.  
- **Metrics Used**: The similarity and distance-based metrics applied.  
- **Visualization**: Graphs and charts to demonstrate OOD detection or drift monitoring.  
- **Results**: Outputs such as flagged OOD instances or charts that show drift detection over time.

### **Example Notebooks**  
| Scenario                         | Notebook Name                         | Link                                                                                 |
|-----------------------------------|---------------------------------------|--------------------------------------------------------------------------------------|
| Within-Modality OOD (Axial)      | `01_axial_vs_non_axial_ct.ipynb`       | [View Notebook](https://github.com/DIDSR/DataMonitor/blob/main/examples/01_axial_vs_non_axial_ct.ipynb) |
| Within-Modality OOD (Image Quality) | `02_image_quality_detection.ipynb`   | [View Notebook](https://github.com/DIDSR/DataMonitor/blob/main/examples/02_image_quality_detection.ipynb) |
| Cross-Modality OOD               | `03_cross_modality_analysis.ipynb`    | [View Notebook](https://github.com/DIDSR/DataMonitor/blob/main/examples/03_cross_modality_analysis.ipynb) |
| Demographic OOD                  | `04_demographic_detection.ipynb`      | [View Notebook](https://github.com/DIDSR/DataMonitor/blob/main/examples/04_demographic_detection.ipynb) |
| Cross-Dataset OOD                | `05_cross_dataset_detection.ipynb`    | [View Notebook](https://github.com/DIDSR/DataMonitor/blob/main/examples/05_cross_dataset_detection.ipynb) |

---

## **How to Use the Notebooks**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/DIDSR/DataMonitor.git
   cd DataMonitor/examples
