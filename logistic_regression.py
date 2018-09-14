import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
enc = preprocessing.LabelEncoder()

train=pd.read_csv(r'C:\Users\HP\Desktop\Titanic_Data\train.csv')
test=pd.read_csv(r'C:\Users\HP\Desktop\Titanic_Data\test.csv')

train['Age']=train['Age'].fillna(train['Age'].mean)
train.drop(train.columns[[3,10,8]], axis=1, inplace=True)
train.Sex.replace(['male', 'female'], [1, 0], inplace=True)
train.Embarked.replace(['S','Q','C'], [0,1,2], inplace=True)
train['Embarked']=train['Embarked'].fillna(train['Embarked'].median)


test['Age']=test['Age'].fillna(test['Age'].median)
test.drop(test.columns[[2,9,7]], axis=1, inplace=True)
test.Sex.replace(['male', 'female'], [1, 0], inplace=True)
test.Embarked.replace(['S','Q','C'], [0,1,2], inplace=True)
test['Embarked']=test['Embarked'].fillna(test['Embarked'].median)
test['Fare']=test['Fare'].fillna(test['Embarked'].median)

model=MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(12,6), random_state=1,activation="relu")

#
#print(train['Age'],train["Embarked"])
#train.Age = train.Age.astype('float64')
#pd.to_numeric(train['Age'],downcast='float',errors='coerce')
#train[['Age','Embarked']] = train[['Age','Embarked']].apply(pd.to_numeric)
#train[['Age','Embarked']] = train[['Age','Embarked']].astype('int64')
#train.apply(pd.to_numeric,  errors='coerce')

# conversions have to be done columns wise Labelencoder really important
#  converting datatype from object to string because object datatype contains many different types of data
train.iloc[:,4]=enc.fit_transform(train.iloc[:,4].astype(str))
# converting strings to floats columns wise
train.iloc[:,4]=enc.fit_transform(train.iloc[:,4])

train.iloc[:,8]=enc.fit_transform(train.iloc[:,8].astype(str))
# converting strings to floats columns wise
train.iloc[:,8]=enc.fit_transform(train.iloc[:,8])

row,col=train.shape

model.fit(train.iloc[:,2:],train.iloc[:,1])

test.iloc[:,3]=enc.fit_transform(test.iloc[:,3].astype(str))
# converting strings to floats columns wise
test.iloc[:,3]=enc.fit_transform(test.iloc[:,3])

test.iloc[:,6]=enc.fit_transform(test.iloc[:,6].astype(str))
# converting strings to floats columns wise
test.iloc[:,6]=enc.fit_transform(test.iloc[:,6])



ans=model.predict(test.iloc[:,1:])
print(ans,type(ans),ans.shape)
k=test.iloc[:,0]
ans=pd.Series(ans)
#k=pd.DataFrame(data=dict(PassengerId=k, Survived=ans), index=k.index)
#print(k)
#k.to_csv('example4.csv')
