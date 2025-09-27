# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 18:28:38 2025

@author: marcl

Name: Marc Lane
Student ID: 21364269/2
Email: lanem2@tcd.ie
"""



### Part (a) (i)



# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Reading in the data
df = pd.read_csv(r"week2.csv", header=None, skiprows=1)
#print(df.head())
X1=df.iloc[:,0].to_numpy()
X2=df.iloc[:,1].to_numpy()
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

# Initializing empty lists
X1_plus = []
X2_plus = []
X1_minus = []
X2_minus = []

# Sorting all data into +1 or -1
for i in range(len(y)):
    if y.iloc[i] == 1: # if y = +1
        X1_plus.append(X1[i])
        X2_plus.append(X2[i])
        
    else:  # if y = -1
        X1_minus.append(X1[i])
        X2_minus.append(X2[i])

# Plotting the raw data
plt.figure(figsize=(8, 6))
plt.scatter(X1_plus, X2_plus, marker='+', color='green', label='+1') # plotting +1 points
plt.scatter(X1_minus, X2_minus, marker='_', color='blue', label='-1') # plotting -1 points
plt.xlabel('x_1', fontsize=20)
plt.ylabel('x_2', fontsize=20)
plt.title('(a) Raw Data', fontsize=20)
plt.legend(loc='lower right', fontsize=15)
plt.tight_layout()
plt.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1, labelsize=12)
plt.show()



### Part (a) (ii)



# Training a logistic regression classifier on the data
lr_model = LogisticRegression()
lr_model.fit(X, y) # .fit() takes matrix X with the two features as its columns, and the y vector
print("")
print("Logistic regression parameter values: intercept = %f, coefficients = %f, %f" % (lr_model.intercept_[0], lr_model.coef_[0][0], lr_model.coef_[0][1])) # printing the parameter values



### Part (a) (iii)



# .predict() uses the logistic regression model to predict the value of y for each row in x 
lr_y_pred = lr_model.predict(X)

# Initialising empty lists
X1_pred_plus = []
X2_pred_plus = []
X1_pred_minus = []
X2_pred_minus = []

# Sorting all prediction data into +1 or -1
for i in range(len(lr_y_pred)):
    if lr_y_pred[i] == 1:
        X1_pred_plus.append(X1[i])
        X2_pred_plus.append(X2[i])
    else:
        X1_pred_minus.append(X1[i])
        X2_pred_minus.append(X2[i])

plt.figure(figsize=(8, 6))

# Distinguishing the trained and predicted data (by highlighting misclassified points)
misclassified = (y != lr_y_pred) # marking points where the predicted data fails to return the trained data
plt.scatter(X1[misclassified], X2[misclassified], marker='x', color='red', s=100, linewidth=1, label='Misclassified')

plt.scatter(X1_pred_plus, X2_pred_plus, marker='+', color='green', s=60, label='+1') # Plotting +1 points
plt.scatter(X1_pred_minus, X2_pred_minus, marker='+', color='blue', s=60, label='-1') # Plotting -1 points

# Creating the decision boundary line of the logistic regression classifier
x1_line = np.linspace(X1.min(), X1.max()) 
x2_line = -(lr_model.intercept_[0] + lr_model.coef_[0][0] * x1_line) / lr_model.coef_[0][1]

# Plotting decision boundary line
plt.plot(x1_line, x2_line, 'r-', label='Decision Boundary')
plt.xlabel('x_1', fontsize=20)
plt.ylabel('x_2', fontsize=20)
plt.legend(loc='lower right', fontsize=15)
plt.title('(a) Logistic Regression Classifier', fontsize=20)
plt.tight_layout()
plt.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1, labelsize=12)
plt.show()



### Part (b) (i) and (ii)



# Testing 3 different penalty parameter values
C_values = [0.001, 1, 100]

for c in C_values:
    
    # Training the linear SVM classifier
    svm_model = LinearSVC(C=c, dual='auto')
    svm_model.fit(X, y) # .fit() takes matrix X with the two features as its columns, and the y vector
    
    # .predict() uses the SVM model to predict the value of y for each row in x 
    y_pred_svm = svm_model.predict(X)
    
    # Printing the parameters
    print("")
    print(f"C = {c} Parameters: intercept = {svm_model.intercept_[0]:.6f}, coefficients = {svm_model.coef_[0][0]:.6f}, {svm_model.coef_[0][1]:.6f}")
    
    # Initialising lists
    X1_pred_plus = []
    X2_pred_plus = []
    X1_pred_minus = []
    X2_pred_minus = []
    
    # Sorting all prediction data into +1 or -1
    for i in range(len(y_pred_svm)):
        if y_pred_svm[i] == 1:
            X1_pred_plus.append(X1[i])
            X2_pred_plus.append(X2[i])
        else:
            X1_pred_minus.append(X1[i])
            X2_pred_minus.append(X2[i])
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Plot predictions
    plt.scatter(X1_pred_plus, X2_pred_plus, marker='+', color='green', s=60, label='+1') # Plotting +1 points
    plt.scatter(X1_pred_minus, X2_pred_minus, marker='+', color='blue', s=60, label='-1') # Plotting -1 points
    
    # Highlighting misclassifications
    misclassified = (y != y_pred_svm)
    plt.scatter(X1[misclassified], X2[misclassified], marker='x', color='red', s=100, linewidth=1, label='Misclassified')
    
    # Creating decision boundary line
    x1_line = np.linspace(X1.min(), X1.max(), 100)
    x2_line = -(svm_model.intercept_[0] + svm_model.coef_[0][0] * x1_line) / svm_model.coef_[0][1]
    plt.plot(x1_line, x2_line, 'r-', linewidth=2, label='Decision Boundary') # plotting decision boundary
    
    plt.xlabel('x_1', fontsize=20)
    plt.ylabel('x_2', fontsize=20)
    plt.title(f'(b) SVM (C = {c})', fontsize=20)
    plt.legend(loc='lower right', fontsize=15)
    plt.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1, labelsize=12)
    plt.tight_layout()
    plt.show()



### Part (c) (i)



# Defining the two additional features
X1_squared = X1**2
X2_squared = X2**2
X_poly = np.column_stack((X1, X2, X1_squared, X2_squared))

# Training logistic regression classifier with 4 features
poly_model = LogisticRegression()
poly_model.fit(X_poly, y)

# Print parameters
print("")
print(" Logistic Regression Parameters:")
print(f" Intercept: {poly_model.intercept_[0]:.6f}")
print(f" Coefficient 1 (x_1): {poly_model.coef_[0][0]:.6f}")
print(f" Coefficient 2 (x_2): {poly_model.coef_[0][1]:.6f}")
print(f" Coefficient 3 (x_1²): {poly_model.coef_[0][2]:.6f}")
print(f" Coefficient 4 (x_2²): {poly_model.coef_[0][3]:.6f}")

# Printing model
print(f"\nPolynomial Logistic Regression model:")
print(f"sign({poly_model.intercept_[0]:.6f} + {poly_model.coef_[0][0]:.6f}*x₁ + {poly_model.coef_[0][1]:.6f}*x₂ + {poly_model.coef_[0][2]:.6f}*x₁² + {poly_model.coef_[0][3]:.6f}*x₂²)")



### Part (c) (ii)



# .predict() uses the logistic regression model to predict the value of y for each row in x 
y_pred_poly = poly_model.predict(X_poly)

# Initializing lists
X1_pred_plus = []
X2_pred_plus = []
X1_pred_minus = []
X2_pred_minus = []

# Sorting all prediction data into +1 or -1
for i in range(len(y_pred_poly)):
    if y_pred_poly[i] == 1:
        X1_pred_plus.append(X1[i])
        X2_pred_plus.append(X2[i])
    else:
        X1_pred_minus.append(X1[i])
        X2_pred_minus.append(X2[i])

# Create the plot
plt.figure(figsize=(10, 8))

# Plotting predictions
plt.scatter(X1_pred_plus, X2_pred_plus, marker='+', color='green', s=60, label='Pred +1')
plt.scatter(X1_pred_minus, X2_pred_minus, marker='+', color='blue', s=60, label='Pred -1')

# Highlight misclassifications
misclassified = (y != y_pred_poly)
plt.scatter(X1[misclassified], X2[misclassified], marker='x', color='red', s=100, linewidth=2, label='Misclassified')



### Part (c) (iv)



# Creating decision boundary line
x1_range = np.linspace(X1.min(), X1.max(), 100)
x2_range = np.linspace(X2.min(), X2.max(), 100)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

# Creating contour plot of decision boundary line
grid_flat = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
grid_poly = np.column_stack([grid_flat, grid_flat**2])  # [x1, x2, x1^2, x2^2]
Z = poly_model.decision_function(grid_poly).reshape(X1_grid.shape) # .decision_function calculates raw decision values (before taking sign) for each grid point
plt.contour(X1_grid, X2_grid, Z, levels=[0], colors='red', linewidths=2) # Draws contour line where Z=0
plt.plot([], [], 'r-', linewidth=2, label='Decision Boundary') 

plt.title('(c) Logistic Regression Classifier with Additional Features', fontsize=20)
plt.xlabel('x_1', fontsize=20)
plt.ylabel('x_2', fontsize=20)
plt.legend(loc='lower right', fontsize=15)
plt.tick_params(direction='in', which='both', top=True, right=True, length=5, width=1, labelsize=12)
plt.tight_layout()
plt.show()



### Part (c) (iii)



# Counting class distribution
total_points = len(y)
num_minus_one = np.sum(y == -1)
num_plus_one = np.sum(y == 1)
most_common_class = -1 if num_minus_one > num_plus_one else 1

# Comparison with baseline (predict most common class)
baseline_accuracy = np.sum(y == most_common_class) / len(y) * 100
print("")
baseline_correct = np.sum(y == most_common_class)
print(f"Baseline (for always predicting {most_common_class}): {baseline_accuracy:.1f}%  ({baseline_correct}/{total_points} correct)")

# Counting misclassifications 
num_misclassified = np.sum(misclassified)
accuracy = (total_points - num_misclassified) / total_points * 100
print(f"\nAccuracy: {accuracy:.1f}% ({total_points - num_misclassified}/{total_points} correct)")













