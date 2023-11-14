# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload the csv file and read the dataset.
3.Check for any null values using the isnull() function.
4. From sklearn.tree inport DecisionTreeRegressor.
5. Import metrics and calculate the Mean squared error.
6. Apply metrics to the dataset, and predict the output.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: R Guruprasad
RegisterNumber: 212222240033
*/
```
```
import pandas as pd
df=pd.read_csv("Salary.csv")
df

df.head()

df.info()

df.isnull().sum()

df['Salary'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head()

X=df[['Position','Level']]
Y=df[['Salary']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/ceafca36-180f-4e46-a1a3-52cf71d2af04)
## df.head()
![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/d209429f-1cb2-4018-b0c1-f04967724896)
## df.info()
![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/075401cf-a5e4-4ffd-9299-56db1e4435d0)
## isnull() & sum() function
![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/661d6962-d344-4adb-b301-ae193eefd166)
![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/0ef11d6b-ddb6-49fd-8305-8fe0f4efe58f)
## data.head() for position
![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/3f6a282f-fc0e-4bbb-90e9-e03a908bfa51)
![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/6843dc03-956b-4c26-b0d0-9e81b04ebcf8)
## MSE value
![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/e1496cb0-1c9f-4e6f-bb80-49f3293a4ed9)
## R2 value
![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/7ee0c604-0fbb-4bed-b408-ad0df3c96f93)
## Prediction value
![image](https://github.com/R-Guruprasad/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119390308/faa3c7f1-6e52-4912-ac47-67edd2dfc19d)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
