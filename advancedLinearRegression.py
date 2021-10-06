# 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥
# https://realpython.com/linear-regression-in-python/

# Import packages and classes
import numpy as np
import statsmodels.api as sm

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

# Add the column of ones to the inputs to be able to calculate the intercept 𝑏₀
x = sm.add_constant(x)
print("x") if isDebugMode else ""
print(x) if isDebugMode else ""

# Create a regression model (OLS = Ordinary Least Squares)
model = sm.OLS(y, x)

# Fit the model with existing data
results = model.fit()


# Get the results of the model fitting
print(results.summary())

# 𝑅²
print('coefficient of determination:', results.rsquared)

# Adjusted 𝑅²
print('adjusted coefficient of determination:', results.rsquared_adj)

#print('intercept:', model.intercept_)
# intercept: 𝑏₀ and coefficients: 𝑏₁ and  𝑏₂ 
print('regression coefficients:', results.params)

# Apply the model for predictions (y = 𝑏₀ + 𝑏₁ * x)
# y_pred = model.intercept_ + model.coef_ * x
y_pred = results.predict(x)
print('predicted response:', y_pred, sep='\n')

# --------
x_new = sm.add_constant(np.arange(10).reshape((-1, 2)))
print(x_new)

y_new = results.predict(x_new)
print(y_new)

