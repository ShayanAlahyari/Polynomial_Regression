import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values  # Extracting independent variable (Position level)
y = dataset.iloc[:, -1].values    # Extracting dependent variable (Salary)

# Split dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train linear regression model on the entire dataset
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

# Transform features to polynomial features of degree 4
poly_reg = PolynomialFeatures(degree=4)
x_poly_degree_4 = poly_reg.fit_transform(x)

# Train polynomial regression model on the transformed features
poly_regressor_degree_4 = LinearRegression()
poly_regressor_degree_4.fit(x_poly_degree_4, y)

# Plot the results of linear and polynomial regression
plt.scatter(x, y, color='red')  # Actual data points
plt.plot(x, linear_regressor.predict(x), color='blue')  # Linear regression line
plt.plot(x, poly_regressor_degree_4.predict(x_poly_degree_4), color='orange')  # Polynomial regression line
plt.title("Salary based on position")
plt.xlabel('Position level')
plt.ylabel("Salary")
plt.show()

# Predict salary for a position level of 6.5 using both models
print(linear_regressor.predict([[6.5]]))
print(poly_regressor_degree_4.predict(poly_reg.fit_transform([[6.5]])))
