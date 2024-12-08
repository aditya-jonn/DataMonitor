"""
Script for extracting radiomics and texture features from general radiological images (e.g., CT, CXR, mammograms).
"""
import SimpleITK as sitk  # Medical image processing
import logging
import radiomics  # PyRadiomics library for feature extraction
from radiomics import featureextractor
import numpy as np
import os
import argparse
from os.path import join as opj
from skimage import morphology, measure
from skimage.filters import gaussian
from scipy import ndimage

# Function to get the largest connected component in a binary mask
def getLargestCC(im_in):
    """
    Extracts the largest connected component from a binary mask.
    - Assumes at least one connected component exists.
    """
    labels = measure.label(im_in)  # Label connected components
    assert labels.max() != 0  # Ensure there is at least one component
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

# Function to generate a binary mask from an input image
def get_mask(im_in):
    """
    Generates a binary mask from an input image.
    - Applies Gaussian smoothing, hole-filling, and morphological opening.
    - Returns the largest connected component as the final mask.
    """
    im_in = np.uint8(gaussian(im_in, sigma=1) * 255)  # Apply Gaussian smoothing
    binary = im_in > 0  # Threshold the image (adjust if needed per dataset)
    binary_filled = ndimage.binary_fill_holes(binary)  # Fill holes in the binary mask
    binary_clean = morphology.opening(binary_filled, morphology.disk(10))  # Morphological opening
    binary_largest = (getLargestCC(binary_clean) * 1).astype(np.uint8)  # Keep the largest connected component
    area_out = np.count_nonzero(binary_largest) / (binary_largest.shape[0] * binary_largest.shape[1])
    return binary_largest, area_out

# Command-line argument parser
parser = argparse.ArgumentParser(description='Feature extraction from radiological images')
parser.add_argument('--path', type=str, help='(str) Folder containing images')
parser.add_argument('--imtype', type=str, help='(str) Image type/dataset name (e.g., CXR, CT, mammograms)')
parser.add_argument('--start', type=int, help='(int) Start index for images to process')
parser.add_argument('--end', type=int, help='(int) End index for images to process')
args = parser.parse_args()

# Input folder and image type
PATH = args.path
imtype = args.imtype
start = args.start
end = args.end

# Output folder for extracted features
SAVE = '/scratch/radiomics_features'
os.makedirs(SAVE, exist_ok=True)

# Radiomics extractor settings
settings = {
    'binCount': 32,  # Number of intensity bins for feature calculation
    'resampledPixelSpacing': None,  # Use original spacing (modify for resampling if needed)
    'interpolator': sitk.sitkLinear,
    'sigma': [1, 2],  # Sigma values for Gaussian-based features
    'distances': [1, 2, 3, 5, 7],  # Distances for texture features
}
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableAllImageTypes()  # Enable all image types for feature extraction

# Suppress logging from PyRadiomics
logger = radiomics.logger
logger.setLevel(logging.CRITICAL)

# Output file for extracted features
output_file = opj(SAVE, f'features-{imtype}_{start:06d}-{end:06d}.txt')
dump = open(output_file, 'w')

# Process images in the specified range
path_list = sorted(os.listdir(PATH))
count = 0
for fn in path_list[start:end]:
    if fn.endswith('.png'):  # Only process PNG images (modify for other formats)
        fnpath = opj(PATH, fn)
        # Read image and convert to SimpleITK format
        im = sitk.ReadImage(fnpath)
        arr = sitk.GetArrayFromImage(im)  # Convert to NumPy array
        
        # Intensity normalization
        arr = np.where(arr > np.percentile(arr, 99.5), np.percentile(arr, 99.5), arr)  # Clip high intensities
        arrn = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255  # Normalize to 0-255
        im = sitk.GetImageFromArray(arrn.astype(np.uint8))  # Convert back to SimpleITK image
        
        # Generate mask
        mask, mask_area = get_mask(arr)
        
        # Extract radiomics features
        featureVector = extractor.execute(im, sitk.GetImageFromArray(mask))
        
        # Write header (only once for the first image)
        if count == 0:
            all_keys = [
                key for key in featureVector.keys() 
                if not (key.startswith('diagnostics') or '3D' in key)  # Exclude 3D features and diagnostics
            ]
            all_keys += ['ds_path', 'ds_name', 'ds_maskarea']
            all_keys = sorted(all_keys)
            
            dump.write(','.join(all_keys) + '\n')  # Write CSV header
        
        # Write feature values
        values = [
            str(featureVector.get(key, '')) if key in featureVector else 
            (fnpath if key == 'ds_path' else 
             fn if key == 'ds_name' else 
             mask_area if key == 'ds_maskarea' else '')
            for key in all_keys
        ]
        dump.write(','.join(values) + '\n')
        
        count += 1

# Close the output file
dump.close()
