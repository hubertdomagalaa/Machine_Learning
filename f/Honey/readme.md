# Honey Production Analysis

## Overview
This project explores the trends in honey production across different states in the United States over several years. Using a dataset that includes information on the total production of honey, the year of production, and the state, we perform an in-depth analysis to understand how honey production has changed over time and identify any patterns or significant changes.

## Data Source
The dataset used in this project is `honeyproduction.csv`, which contains the following columns:
- `state`: The state in the US where the honey was produced.
- `year`: The year of honey production.
- `totalprod`: The total production of honey (in pounds).
- Additional columns related to production specifics and economic factors.

## Tools and Libraries
The project utilizes Python as the primary programming language with the following libraries:
- `pandas` for data manipulation and analysis.
- `matplotlib` and `seaborn` for data visualization.
- `scikit-learn` for implementing machine learning models to predict future trends in honey production.

## Project Structure
1. **Data Loading and Preprocessing**: The dataset is loaded into a pandas DataFrame, and preliminary data cleaning steps are performed to prepare the data for analysis.

2. **Data Exploration**: We perform exploratory data analysis (EDA) to understand the dataset better. This includes visualizing the distribution of honey production across different states and years.

3. **Feature Engineering**: New features are created to aid in the analysis, such as the percentage change in honey production from the previous year.

4. **Aggregating Data**: The data is aggregated to calculate the mean total production and mean production change per year.

5. **Modeling**: A linear regression model is trained to predict future honey production based on the year. The dataset is split into training and testing sets to evaluate the model's performance.

6. **Evaluation**: The model's performance is evaluated using metrics such as the mean squared error (MSE) and R-squared value.

## Key Findings
- The analysis of honey production trends over the years.
- Insights into how production varies by state.
- Predictions for future honey production trends.

