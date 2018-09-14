import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
enc = preprocessing.LabelEncoder()


dt=pd.read_csv(r"C:\Users\HP\Desktop\mushroom.csv",header=None)
row,col=dt.shape

# conversions have to be done columns wise
for i in range(col):
    # converting datatype from object to string because object datatype contains many datatypes
    dt.iloc[:,i]=enc.fit_transform(dt.iloc[:,i].astype(str))
    # converting strings to floats columns wise
    dt.iloc[:,i]=enc.fit_transform(dt.iloc[:,i])

train,test=train_test_split(dt,test_size=.20,random_state=False)


#target is the first column
train_set=train.iloc[0:,1:]
train_target=train.iloc[0:,0:1]
test_set=train.iloc[0:,1:]
test_target=train.iloc[0:,0:1]

print(train_target.shape,train_set.shape,type(train_set))
model=RandomForestClassifier(n_estimators=40)
#train_target=list(train_target)
#train_target=np.transpose(train_target)
model.fit(train_set,np.ravel(train_target,order='C'))
model.predict(test_set)
#test_target=np.ravel(test_target,order='C')
print(test_set," ",test_target)
print(model.score(test_set,test_target)*100)