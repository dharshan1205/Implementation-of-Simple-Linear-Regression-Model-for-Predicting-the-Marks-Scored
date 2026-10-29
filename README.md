# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Create a dataset containing study hours and marks scored by students.

3.Separate the study hours as the input variable (X) and marks as the output variable (Y).

4.Split the dataset into training and testing data.

5.Create a Simple Linear Regression model.

6.Train the model using the training data.

7.Predict the marks using the testing data.

8.Get the study hours from the user.

9.Predict and display the marks scored by the student.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHARSHAN R
RegisterNumber: 212224230060
*/
```
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

```
```
df = pd.read_csv('exp_2_dataset_student_scores.csv')   
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
```

<img width="317" height="317" alt="image" src="https://github.com/user-attachments/assets/7c47e3ad-fe82-4915-b672-4f1d93829c07" />

```
X = df.iloc[:, :-1].values   
Y = df.iloc[:, -1].values    
print("X (features):", X.flatten())
print("Y (targets):", Y)
```

<img width="807" height="106" alt="image" src="https://github.com/user-attachments/assets/9261ef8b-dfdf-43d4-802c-aae5a1bbc491" />

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))
```

<img width="431" height="33" alt="image" src="https://github.com/user-attachments/assets/1f7a2b15-ec3c-430e-b3c8-31c339bda15b" />

```
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```

<img width="331" height="90" alt="image" src="https://github.com/user-attachments/assets/76166652-c508-4cd5-bcfc-5f04c802dd2a" />

```
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```

<img width="681" height="495" alt="image" src="https://github.com/user-attachments/assets/7c5a5ca3-6f2f-48e6-aeb4-9b55ad960c6a" />


```
Y_pred = regressor.predict(X_test)
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```

<img width="672" height="462" alt="image" src="https://github.com/user-attachments/assets/e810c6c4-d0b7-4003-8d7e-9f766be35f48" />

```
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
```
<img width="487" height="83" alt="image" src="https://github.com/user-attachments/assets/d89e0fd5-9ace-43b7-89a6-608e508c4de8" />

```
new_hours = np.array([[2.5], [8.0]])  
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
```

<img width="526" height="30" alt="image" src="https://github.com/user-attachments/assets/146d318a-7e84-41dc-b1f3-2940e96586cd" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
