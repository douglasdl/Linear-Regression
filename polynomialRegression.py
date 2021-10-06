# 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥 + 𝑏₂𝑥²
# https://realpython.com/linear-regression-in-python/

# Import packages and classes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

isDebugMode = True

# Provide data (x, y)
# input = regressor: x
x = np.array([5, 15, 25, 35, 45, 55])
print("x") if isDebugMode else ""
print(x) if isDebugMode else ""

# Transform x in a one column matrix
x = x.reshape((-1, 1))
print("x") if isDebugMode else ""
print(x) if isDebugMode else ""

# output = predictor: y
y = np.array([15, 11, 2, 8, 25, 32])
print("y") if isDebugMode else ""
print(y) if isDebugMode else ""

# Transform input data
transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)

# 1st column is the original x and the 2nd column is 𝑥²
x_ = transformer.transform(x)
print(x_)

# Create a regression model
model = LinearRegression()

# Fit the model with existing data
model.fit(x_, y)

# Get the results (𝑅²) of the model fitting
r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)

# 𝑏₀
print('intercept:', model.intercept_)

# 𝑏₁ and 𝑏₂
print('coefficients:', model.coef_)

# Apply the model for predictions (y = 𝑏₀ + 𝑏₁ * x)
# y_pred = model.intercept_ + model.coef_ * x
y_pred = model.predict(x_)
print('predicted response:', y_pred, sep='\n')