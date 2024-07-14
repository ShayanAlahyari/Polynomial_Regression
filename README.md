# Polynomial Regression

This repository contains code for performing polynomial regression on a dataset that includes position levels and corresponding salaries. The goal is to predict the salary based on the position level using both linear and polynomial regression models.

## Dataset

The dataset used is `Position_Salaries.csv`, which consists of two columns:
- Position level
- Salary

## Code Description

The provided code performs the following steps:

1. **Import necessary libraries**:
   - `numpy`, `pandas`, `matplotlib` for data manipulation and visualization
   - `sklearn.model_selection` for splitting the dataset
   - `sklearn.linear_model` for creating linear regression model
   - `sklearn.preprocessing` for transforming features to polynomial features

2. **Load and preprocess the dataset**:
   - Load the dataset using `pandas`
   - Extract the independent variable (Position level) and dependent variable (Salary)

3. **Split the dataset**:
   - Split the dataset into training and test sets using `train_test_split`

4. **Train linear regression model**:
   - Train a linear regression model on the entire dataset

5. **Train polynomial regression model**:
   - Transform features to polynomial features of degree 4
   - Train a polynomial regression model on the transformed features

6. **Plot the results**:
   - Plot the actual data points, linear regression line, and polynomial regression line

7. **Predict salary for a specific position level**:
   - Predict salary for a position level of 6.5 using both linear and polynomial regression models

## Results

The code outputs predictions for the salary at position level 6.5 using both linear and polynomial regression models. Additionally, it visualizes the actual data points and the regression lines for both models.

## How to Run

1. Ensure you have the necessary libraries installed:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
