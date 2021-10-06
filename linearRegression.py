# 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥
# https://realpython.com/linear-regression-in-python/

# Import packages and classes
import numpy as np
from sklearn.linear_model import LinearRegression

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
y = np.array([5, 20, 14, 32, 22, 38])

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

# 𝑏₁ 
print('slope:', model.coef_)

# Apply the model for predictions (y = 𝑏₀ + 𝑏₁ * x)
# y_pred = model.intercept_ + model.coef_ * x
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

# --------
x_new = np.arange(5).reshape((-1, 1))
print(x_new)

y_new = model.predict(x_new)
print(y_new)

