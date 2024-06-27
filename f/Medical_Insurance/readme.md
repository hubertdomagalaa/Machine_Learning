# Medical Insurance Cost Prediction

## Overview
This Python project aims to predict the medical insurance cost for individuals based on various factors such as age, sex, BMI (Body Mass Index), number of children, and smoking status. It utilizes object-oriented programming principles to model patient information and calculate estimated insurance costs.

## Features
- **Patient Profile Management**: Allows for the creation and updating of patient profiles, including personal and health-related information.
- **BMI Calculation**: Computes the BMI based on the patient's height and weight.
- **Insurance Cost Estimation**: Estimates the insurance cost using a formula that considers age, sex, BMI, number of children, and smoking status.
- **Data Validation**: Ensures that all patient information provided is within realistic and acceptable ranges.

## Classes and Enums
- `Sex`: An enumeration to represent the sex of a patient (Male or Female).
- `SmokingStatus`: An enumeration to represent the smoking status of a patient (Smoker or Non-Smoker).
- `Patient`: A class that encapsulates all patient-related information and methods for managing patient data and calculating insurance costs.

## Usage
To use this project, instantiate a `Patient` object with the required parameters (name, age, sex, height, weight, number of children, smoker status). After creating a patient object, you can:
- Update patient information (age, number of children, smoking status).
- Calculate the patient's BMI.
- Estimate the insurance cost.
- Retrieve the patient's profile.

### Example
```python
from medical_insurance import Patient, Sex, SmokingStatus

try:
    patient1 = Patient("John Doe", 25, Sex.MALE, 175, 70, 0, SmokingStatus.NON_SMOKER)
    print(patient1.patient_profile())
    patient1.estimated_insurance_cost()
    patient1.update_age(26)
    patient1.update_num_children(1)
    patient1.update_smoking_status(SmokingStatus.SMOKER)
except ValueError as e:
    print(f"Error: {e}")