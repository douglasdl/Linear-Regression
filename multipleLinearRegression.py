# 𝑓(𝑥₁, 𝑥₂) = 𝑏₀ + 𝑏₁𝑥₁ + 𝑏₂𝑥₂
# https://realpython.com/linear-regression-in-python/

# Import packages and classes
import numpy as np
from sklearn.linear_model import LinearRegression

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

# Create a regression model
model = LinearRegression()

# Fit the model with existing data
model.fit(x, y)

# Get the results (𝑅²) of the model fitting
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

# 𝑏₀
print('intercept:', model.intercept_)

# 𝑏₁ and 𝑏₂
print('slope:', model.coef_)

# Apply the model for predictions (y = 𝑏₀ + 𝑏₁𝑥₁ + 𝑏₂𝑥₂)
# y_pred = model.intercept_ + model.coef_ * x
# y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

# --------
x_new = np.arange(10).reshape((-1, 2))
print(x_new)

y_new = model.predict(x_new)
print(y_new)

