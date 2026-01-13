# Income Classification Project

## Overview
This project tackles the binary classification problem of predicting whether an individual's annual income exceeds $50,000 based on demographic and employment-related features. The dataset is derived from the 1994 US Census and is commonly used for demonstrating classification algorithms and fairness in machine learning.

## Problem Statement
**Goal:** Predict income bracket (≤$50K or >$50K) using socioeconomic and demographic information.

**Business Value:** Understanding income predictors can inform:
- Economic policy decisions
- Targeted social programs
- Marketing and customer segmentation
- Fairness and bias analysis in ML systems

## Dataset
**Source:** UCI Machine Learning Repository - Adult Income Dataset

**Size:** 48,842 instances (32,561 training + 16,281 test)

### Features (14 attributes)
**Continuous Features:**
1. **age:** Age in years
2. **fnlwgt:** Final sampling weight (represents population count)
3. **education-num:** Years of education
4. **capital-gain:** Income from investment sources
5. **capital-loss:** Losses from investment sources
6. **hours-per-week:** Average hours worked per week

**Categorical Features:**
7. **workclass:** Employment sector (Private, Self-emp, Gov, etc.)
8. **education:** Highest education level (Bachelors, HS-grad, Masters, etc.)
9. **marital-status:** Marital status (Married, Single, Divorced, etc.)
10. **occupation:** Type of occupation (Tech-support, Sales, Exec-managerial, etc.)
11. **relationship:** Family relationship (Husband, Wife, Own-child, etc.)
12. **race:** Racial category
13. **sex:** Gender (Male, Female)
14. **native-country:** Country of origin

**Target Variable:** income (≤50K, >50K)

## Methodology

### 1. Data Preprocessing
- Handle missing values (denoted as "?")
- Encode categorical variables (One-Hot Encoding or Label Encoding)
- Feature scaling for continuous variables
- Address class imbalance (income >$50K is minority class ~24%)

### 2. Exploratory Data Analysis
- Distribution analysis of features by income class
- Correlation analysis between features
- Visualize relationships (age vs. income, education vs. income, etc.)
- Identify potential bias in sensitive attributes (race, sex)

### 3. Feature Engineering
- Create derived features:
  - **education_occupation:** Combined feature for job-education match
  - **capital_net:** capital-gain minus capital-loss
  - **age_bins:** Categorized age groups
- Remove redundant features (e.g., education and education-num)
- Feature selection using correlation and feature importance

### 4. Model Training
Algorithms evaluated:
- Logistic Regression (baseline)
- Decision Trees
- Random Forest Classifier
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines
- Neural Networks (MLPClassifier)

### 5. Model Evaluation
Metrics used:
- **Accuracy:** Overall correctness
- **Precision & Recall:** For each income class
- **F1-Score:** Balanced metric
- **ROC-AUC:** Classification performance across thresholds
- **Confusion Matrix:** Detailed error analysis

### 6. Fairness Analysis
- Evaluate model performance across demographic groups (race, sex)
- Measure disparate impact and equalized odds
- Investigate potential bias in predictions

## Libraries Used
- **pandas**, **numpy** for data manipulation
- **scikit-learn** for ML models and preprocessing
- **matplotlib**, **seaborn** for visualization
- **xgboost**, **lightgbm** for advanced ensemble methods
- **imbalanced-learn** for handling class imbalance

## Key Results
- **Best Model:** Gradient Boosting Classifier
- **Accuracy:** ~86%
- **ROC-AUC:** ~0.91
- **Most Important Features:**
  1. Marital status (married individuals significantly more likely >$50K)
  2. Education level (higher education correlates with higher income)
  3. Capital gains (strong predictor)
  4. Age (income increases with age up to ~50 years)
  5. Hours per week (longer work hours correlate with higher income)

## Insights
- **Education matters:** Individuals with Bachelor's degree or higher are 3x more likely to earn >$50K
- **Marriage effect:** Married individuals have significantly higher income rates
- **Gender gap:** Model reveals income disparities between male and female workers
- **Occupation impact:** Executive-managerial and professional-specialty occupations strongly associated with higher income

## Ethical Considerations
This dataset contains sensitive demographic information. Key considerations:
- **Fairness:** Models may perpetuate historical biases (e.g., gender pay gap)
- **Privacy:** Even aggregated census data requires responsible handling
- **Use Cases:** Should not be used for discriminatory decision-making
- **Transparency:** Model predictions should be explainable to affected individuals

## Applications
- **Economic Research:** Understanding income determinants
- **Policy Analysis:** Evaluating impact of education and employment policies
- **ML Education:** Teaching classification, fairness, and bias mitigation
- **Customer Segmentation:** Marketing strategies (with ethical considerations)

## Future Improvements
- Implement fairness-aware algorithms (e.g., fairness constraints)
- Use more recent census data (1994 is outdated)
- Add SHAP values for model interpretability
- Create interactive dashboard for exploring predictions
- A/B testing different fairness mitigation strategies

## Conclusion
This project demonstrates classification techniques on real-world socioeconomic data while highlighting the importance of fairness and bias considerations in machine learning systems. The analysis reveals that education, occupation, marital status, and capital gains are the strongest predictors of income level.
