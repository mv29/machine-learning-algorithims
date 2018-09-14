import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB , BernoulliNB , MultinomialNB

dt= pd.read_csv(r"C:\Users\HP\Desktop\train.csv")


# converting strings to numeric values

#first way adding a new columns according the sex and embarked columns
dt["Sex_cleaned"]=np.where(dt["Sex"]=="male",0,1)
dt["Embarked_cleaned"]=np.where(dt["Embarked"]=="S",0,np.where(dt["Embarked"]=="C",1,np.where(dt["Embarked"]=="Q",2,3)))

del dt['Sex']
del dt['Embarked']

#cleaning the data by dropping rows having NAN
dt=dt[[
    "Survived",
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]].dropna(axis=0, how='any')
print(dt['Survived'],type(dt['Survived']))

_train,x_test=train_test_split(dt,test_size=0.5,random_state=int(time.time()))
print(x_test,x_test)

model=GaussianNB()
model.fit(x_train.iloc[:,1:],x_train.iloc[:,0])
result=model.predict(x_test.iloc[:,1:])
print(model.score(result.reshape(-1,1),x_test.iloc[:,0])*100)

