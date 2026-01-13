# Raisin Classification Project

## Overview
This project focuses on classifying two varieties of raisins (Kecimen and Besni) grown in Turkey using machine learning classification algorithms. The dataset contains morphological features extracted from images of raisins using computer vision techniques.

## Dataset
The Raisin Dataset includes 900 raisin grains:
- **Kecimen variety:** 450 samples
- **Besni variety:** 450 samples

### Features (7 morphological characteristics)
1. **Area:** Number of pixels within the boundaries of the raisin
2. **Perimeter:** Total length of the boundary (in pixels)
3. **MajorAxisLength:** Length of the main axis of the ellipse fitted to the raisin
4. **MinorAxisLength:** Length of the minor axis of the ellipse
5. **Eccentricity:** Measure of how elongated the ellipse is
6. **ConvexArea:** Area of the smallest convex hull containing the raisin
7. **Extent:** Ratio of raisin area to bounding rectangle area

**Target Variable:** Class (Kecimen or Besni)

## Methodology
The project involves:
1. **Data Loading and Exploration:** Understanding feature distributions and class balance
2. **Data Preprocessing:** Feature scaling and normalization for optimal model performance
3. **Exploratory Data Analysis:** Visualizing feature correlations and class separability
4. **Model Training:** Implementing various classification algorithms including:
   - Logistic Regression
   - Support Vector Machines (SVM)
   - Random Forest Classifier
   - K-Nearest Neighbors (KNN)
   - Decision Trees
5. **Model Evaluation:** Comparing performance using accuracy, precision, recall, and F1-score
6. **Feature Importance Analysis:** Identifying which morphological features are most discriminative

## Libraries Used
- **pandas** and **numpy** for data manipulation
- **scikit-learn** for machine learning models and preprocessing
- **matplotlib** and **seaborn** for data visualization
- **scipy** for statistical analysis

## Key Objectives
- Achieve high classification accuracy (>90%)
- Identify the most important morphological features
- Compare performance across different ML algorithms
- Provide insights for automated raisin quality control systems

## Results
The best-performing models achieve **>95% accuracy** in distinguishing between the two raisin varieties, with **MajorAxisLength**, **MinorAxisLength**, and **Eccentricity** being the most important features for classification.

## Applications
- **Agricultural Quality Control:** Automated sorting of raisin varieties
- **Food Industry:** Quality assurance and product standardization
- **Computer Vision:** Demonstrating image-based feature extraction for classification

## Conclusion
This project demonstrates the effectiveness of machine learning in agricultural classification tasks, showing that morphological features extracted from images can reliably differentiate raisin varieties with high accuracy.
