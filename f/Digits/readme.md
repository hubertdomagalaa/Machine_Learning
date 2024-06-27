# Digits Recognition Project

## Overview
This project focuses on the optical recognition of handwritten digits. It utilizes a dataset containing images of hand-written digits, with 10 classes where each class refers to a digit. The dataset comprises 1797 instances, each with 64 attributes representing an 8x8 image of integer pixels in the range 0 to 16.

## Dataset
The dataset used in this project is a copy of the test set of the UCI ML hand-written digits datasets. Preprocessing programs made available by NIST were used to extract normalized bitmaps of handwritten digits from a preprinted form. The data set contains images of hand-written digits, with each class referring to a digit. The 32x32 bitmaps are divided into nonoverlapping blocks of 4x4, and the number of on pixels are counted in each block. This generates an input matrix of 8x8 where each element is an integer in the range 0 to 16.

## Technologies Used
- Python
- Scikit-learn for machine learning models
- Matplotlib and Seaborn for visualization
- PCA for dimensionality reduction
- Various classification algorithms including SVM, Random Forest, and MLPClassifier

## Features
- Loading and visualizing the digits dataset
- Preprocessing the data
- Splitting the dataset into training and testing sets
- Applying various machine learning models to classify the digits
- Evaluating the models using accuracy score, confusion matrix, and classification report
- Utilizing PCA for dimensionality reduction
- Hyperparameter tuning using GridSearchCV

