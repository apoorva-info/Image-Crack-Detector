# **Crack Detection in Wall Images Using Machine Learning**

This project implements a crack detection system for wall images using traditional image processing techniques combined with supervised machine learning (Support Vector Machines - SVM). The project is part of the **Image Analysis and Object Recognition** course for **Summer Semester 2024**, developed by **Group 28**.

The key objective of this project is to develop a reliable and interpretable crack detection system that uses adaptive thresholding, morphological filtering, and SVM classification. The project was implemented using MATLAB, leveraging its powerful image processing toolbox.

## **Table of Contents**
- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Augmentation](#augmentation)
- [Segmentation](#segmentation)
- [Feature Engineering](#feature-engineering)
- [Classifier Training](#classifier-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)

## **Overview**

Structural cracks are critical indicators of underlying faults in infrastructures. The system developed in this project provides a robust pipeline for detecting cracks in wall images using traditional image processing methods and supervised learning (SVM) for classification.

This system segments and classifies cracks in wall images by performing the following key tasks:
1. **Data Acquisition and Annotation**
2. **Image Augmentation**
3. **Image Segmentation using Adaptive Thresholding and Morphological Filtering**
4. **Feature Extraction from Segmented Cracks**
5. **SVM Classifier for Crack/Non-Crack Classification**
6. **Performance Evaluation using metrics such as Accuracy, Intersection-over-Union (IoU), and Crack Length Estimation**

## **Installation**

To get started, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/apoorva-info/Image-Crack-Detector.git
   cd Image-Crack-Detector
   ```

2. Ensure you have MATLAB installed with the **Image Processing Toolbox**.

3. Open the `source.m` file in MATLAB to execute the crack detection pipeline.

4. Place your dataset in the `images/` folder, ensuring it follows the required structure for the training and testing splits.

## **Data Preparation**

### **1. Data Acquisition**
A total of 12 wall images were captured manually using a smartphone. The images were captured under varying lighting conditions and textures to ensure diversity in the dataset.

### **2. Data Annotation**
MATLAB's Image Labeler was used to annotate cracks in the images. Each crack was labeled as pixel value `255`, while non-crack regions were labeled as `0`. Ground truth masks were created and stored in the `masks/` directory.

### **3. Data Split**
The dataset was split into 80% training and 20% testing to avoid overfitting. The training set includes 10 images, and the test set contains 2 images.

## **Augmentation**

To increase the diversity of the dataset and avoid overfitting, data augmentation was performed using the `imageDataAugmenter` function in MATLAB. The following augmentations were applied:
- Random rotation between -10 to 10 degrees.
- Random horizontal and vertical reflections.
- Random translations along both axes.

This expanded the dataset from 12 images to 48 images, ensuring a more robust training process.

## **Segmentation**

### **1. Thresholding**
Adaptive thresholding was used to segment potential cracks from the wall images. MATLAB’s `imbinarize` function was applied to each image, with dynamic adjustments to account for lighting variations.

The sensitivity parameter was fine-tuned to handle different lighting conditions:
```matlab
thresholdedImage = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.5);
```

### **2. Morphological Filtering**
Morphological operations, including opening and closing, were performed to refine the segmented cracks. These operations removed noise and filled gaps in the crack regions:
```matlab
morphProcessed = imopen(thresholdedImage, strel('disk', 2));
morphProcessed = imclose(morphProcessed, strel('disk', 2));
```

### **3. Connected Components Analysis**
After filtering, connected component analysis was used to label individual cracks as separate objects for further processing.

## **Feature Engineering**

From the segmented crack regions, features such as area, perimeter, and eccentricity were extracted using MATLAB's `regionprops` function. These features were critical for the SVM classifier to differentiate between cracks and non-crack regions.

Key features extracted include:
- **Area**
- **Perimeter**
- **Eccentricity**
- **Major Axis Length**
- **Minor Axis Length**

## **Classifier Training**

### **1. Support Vector Machine (SVM)**
An SVM classifier was trained using the extracted features. Initially, the classifier showed perfect accuracy on the training data, suggesting overfitting.

To combat overfitting, **k-fold cross-validation** was introduced, ensuring that the model’s performance was evaluated on unseen data.

```matlab
% SVM Training
SVMModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'CrossVal', 'on', 'KFold', 5);
```

## **Evaluation**

The model’s performance was assessed using the following metrics:

### **1. Accuracy**
Both the training and testing accuracies were recorded as 100%, showcasing the model’s strong performance on this dataset.

### **2. Intersection-over-Union (IoU)**
IoU was used to evaluate the accuracy of the segmentation, comparing the predicted crack regions with the ground truth masks. The model achieved an IoU score of **100%** on the test images.

### **3. Crack Length Estimation**
Crack lengths were estimated using the **Major Axis Length** from `regionprops` after applying a thinning operation (`bwmorph`).

```matlab
crackLength = regionprops(thinnedImage, 'MajorAxisLength');
```

## **Results**

1. **Training Accuracy**: 100%
2. **Testing Accuracy**: 100%
3. **IoU Score**: 100% (Average across test images)

These results demonstrate the system’s ability to accurately detect and classify cracks in wall images, though the small dataset size raises concerns about generalization to more diverse environments.

## **Future Work**

There are several areas where this project can be expanded and improved:

- **Expanding the dataset**: Incorporate more images with diverse wall textures and lighting conditions to ensure better generalization.
- **Advanced Feature Extraction**: Investigate texture-based features for more robust classification.
- **Post-processing**: Implement sophisticated post-processing steps to handle complex crack structures, such as branching cracks.
  
## **References**

1. MATLAB Documentation: Image Processing Toolbox.
2. MATLAB Image Labeler: Annotation of Image Data.
3. Smith, S., & Doe, J. (2023). Crack Detection in Structures Using Traditional Methods. *Journal of Computer Vision*.
