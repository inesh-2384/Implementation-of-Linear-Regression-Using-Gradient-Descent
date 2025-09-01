# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset.

2. Initialize model parameters.

3. Apply gradient descent to update weights.

4. Train until cost converges.

5. Predict profit using the trained model.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: INESH.N
RegisterNumber: 212223220036

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("50_Startups.csv")
print("First 5 rows of dataset:\n", data.head())

# Preprocessing
# Select features (R&D Spend, Administration, Marketing Spend) and target (Profit)
X = data[["R&D Spend", "Administration", "Marketing Spend"]].values
y = data["Profit"].values.reshape(-1, 1)

# Feature scaling (Normalization for gradient descent)
X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
y_mean, y_std = np.mean(y), np.std(y)

X = (X - X_mean) / X_std
y = (y - y_mean) / y_std

# Add intercept column (bias term)
m = len(y)  # number of samples
X = np.hstack((np.ones((m, 1)), X))

# Gradient Descent Function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = (1/m) * X.T.dot(errors)
        theta = theta - alpha * gradients
        cost = (1/(2*m)) * np.sum(np.square(errors))
        cost_history.append(cost)

    return theta, cost_history

# Initialize variables
theta = np.zeros((X.shape[1], 1))  # Initialize weights
alpha = 0.01   # Learning rate
iterations = 1500

# Train the model using Gradient Descent
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

print("Optimized Parameters (Theta):\n", theta)
print("Final Cost:", cost_history[-1])

# Plot Cost vs Iterations
plt.plot(range(iterations), cost_history, 'b')
plt.xlabel("Iterations")
plt.ylabel("Cost (J)")
plt.title("Convergence of Gradient Descent")
plt.show()

# Prediction Example
# Example: Predict profit for R&D = 160000, Admin = 130000, Marketing = 300000
sample = np.array([[160000, 130000, 300000]])

# Apply same normalization as training data
sample = (sample - X_mean) / X_std
sample = np.hstack((np.ones((1, 1)), sample))  # add intercept

predicted_profit = sample.dot(theta)

# Convert prediction back to original scale
predicted_profit = predicted_profit * y_std + y_mean
print("Predicted Profit for sample input:", predicted_profit[0][0])


```

## Output:
```
Developed by: INESH.N
RegisterNumber: 212223220036
```
### Dataset Preview
<img width="694" height="165" alt="image" src="https://github.com/user-attachments/assets/29180800-df15-461c-8fd2-4fb9abaaed05" />

### Optimized Parameters
<img width="312" height="123" alt="image" src="https://github.com/user-attachments/assets/6f175c7d-22f7-4bd8-beb7-eb38dced6859" />

### Final Cost
<img width="328" height="38" alt="image" src="https://github.com/user-attachments/assets/006027dd-5f8f-4a69-9eea-8ca18eee797f" />

### Convergence Graph
<img width="721" height="584" alt="image" src="https://github.com/user-attachments/assets/c2b9d370-152d-4c2a-84dd-b2f3f15cf403" />

### Prediction Result
<img width="531" height="52" alt="image" src="https://github.com/user-attachments/assets/3dbf57a3-e6c4-4e12-930b-5e1a4f6930a4" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
