# Fraud Detection Using Logistic Regression

## Project Overview
This project focuses on detecting fraudulent transactions within a dataset using Logistic Regression. The goal is to identify patterns and characteristics of transactions that are likely to be fraudulent.

## Dataset
The dataset used in this project contains various features related to transaction details. Some of the key features include:
- `step`: represents a unit of time where 1 step equals 1 hour
- `type`: type of transaction (e.g., PAYMENT, TRANSFER)
- `amount`: transaction amount
- `nameOrig`: customer starting the transaction
- `oldbalanceOrg`: initial balance before the transaction
- `newbalanceOrig`: new balance after the transaction
- `nameDest`: recipient of the transaction
- `oldbalanceDest`: initial recipient balance before the transaction
- `newbalanceDest`: new recipient balance after the transaction
- `isFraud`: indicates if the transaction is fraudulent
- `isFlaggedFraud`: flags illegal attempts to transfer more than 200,000 in a single transaction

## Features Engineering
Several new features were created to aid in the fraud detection analysis:
- `isPayment`: indicates if the transaction is a payment or debit
- `isMovement`: indicates if the transaction involves cash out or transfer
- `transactionRatio`: ratio of the transaction amount to the original balance
- `accountDiff`: absolute difference between the original and destination account balances

## Model
A Logistic Regression model was chosen for its simplicity and effectiveness in binary classification tasks. The model is trained to distinguish between fraudulent and non-fraudulent transactions.

## Evaluation
The model's performance is evaluated using various metrics, including confusion matrix, classification report, and ROC curve. These metrics help in understanding the model's ability to correctly identify fraudulent transactions.

