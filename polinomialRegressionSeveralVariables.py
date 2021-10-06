# 𝑓(𝑥₁, 𝑥₂) = 𝑏₀ + 𝑏₁𝑥₁ + 𝑏₂𝑥₂ + 𝑏₃𝑥₁² + 𝑏₄𝑥₁𝑥₂ + 𝑏₅𝑥₂²
# https://realpython.com/linear-regression-in-python/

# Import packages and classes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

isDebugMode = True

# Provide data (x, y)
# input = regressor: x
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
x = np.array(x)
print("x") if isDebugMode else ""
print(x) if isDebugMode else ""

# output = predictor: y
y = [4, 5, 20, 14, 32, 22, 38, 43]
y = np.array(y)
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

# 𝑏₁, 𝑏₂, 𝑏₃, 𝑏₄ and 𝑏₅
print('coefficients:', model.coef_)

# Apply the model for predictions (y = 𝑏₀ + 𝑏₁𝑥₁ + 𝑏₂𝑥₂ + 𝑏₃𝑥₁² + 𝑏₄𝑥₁𝑥₂ + 𝑏₅𝑥₂²)
# y_pred = model.intercept_ + model.coef_ * x
y_pred = model.predict(x_)
print('predicted response:', y_pred, sep='\n')