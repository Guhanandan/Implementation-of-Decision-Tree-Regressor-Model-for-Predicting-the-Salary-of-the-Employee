# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries from python.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply to the model from the dataset.

5.Predict the values of the arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model from the dataset.

7.Predict the values of array

8.Apply it to the new unknown values.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: GUHANANDAN V
RegisterNumber:  212221220014
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:

![1 (1)](https://user-images.githubusercontent.com/100425381/205338353-edbea02a-98c9-4604-a3cf-4ddf40ec6b22.png)

![2 (1)](https://user-images.githubusercontent.com/100425381/205338371-d9424449-b8aa-4520-8770-7e171ef8cbea.png)

![3 (1)](https://user-images.githubusercontent.com/100425381/205338367-fdb4ad0d-7678-4918-8cf6-68fe7b46d91b.png)

![4 (1)](https://user-images.githubusercontent.com/100425381/205338366-3329a621-9d0e-43f0-8eca-3c95e5b683a5.png)

![5 (1)](https://user-images.githubusercontent.com/100425381/205338362-1b612e7b-9650-407a-b1ed-3f5e2e36df08.png)

![6 (1)](https://user-images.githubusercontent.com/100425381/205338358-c6494930-88b5-4767-8e05-039f5c344e2e.png)

![7](https://user-images.githubusercontent.com/100425381/205338356-02625cb0-dfeb-4ddb-a17a-90b0cf590561.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
