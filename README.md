# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PRAVEENA N
RegisterNumber:  212222040122
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()    #no departments and no left
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
#### data.head()
![Screenshot 2023-06-03 181522](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115707860/28a02272-d076-488b-bc0e-258db61a3b29)
#### data.info()
![Screenshot 2023-06-03 181528](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115707860/cdc55078-656b-49d7-b171-ba2be9a56bfd)
#### isnull() and sum()
![Screenshot 2023-06-03 181534](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115707860/c8005750-fb30-47db-9c4d-6309e48560b7)
#### data value counts()
![Screenshot 2023-06-03 181649](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115707860/8cbea8e5-088d-4896-8c3f-eabcedd9b7a0)
#### data.head() for salary
![Screenshot 2023-06-03 181719](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115707860/7123868e-1b73-448a-a92c-bfcb52c69d22)
#### x.head()
![Screenshot 2023-06-03 181729](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115707860/7900550f-09ae-4b93-baa2-4e1d8cd21967)
#### accuracy value
![Screenshot 2023-06-03 181805](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115707860/d6235dee-d231-4eb2-9540-5a8fcc8d845f)
#### data prediction
![Screenshot 2023-06-03 181815](https://github.com/Yamunaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115707860/79384c60-837a-474e-adc4-cb88724287ce)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
