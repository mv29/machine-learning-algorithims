import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df=pd.read_csv("C:\\Users\\HP\\Desktop\\zoodataset.csv", names=['animal_name','hair','feathers','eggs','milk','airbone','aquatic','predator','toothed','backbone' ,'breathes','venomous','fins','legs','tail','domestic','catsize','class'])


# or df=df.drop('atnimal_name',axis=1)
# all those function having a dot in its function will always return something and will not change the original data
# instead will return a copy of the original data with the desired changes

# deleting column by its name
del df['animal_name']

# most of the functions in pandas have [] syntax instead of {}
#Split the data into a training and a testing set

# way 1 to split
# splitting columns using indexes
train_set=df.iloc[0:80, :-1]
test_set=df.iloc[80:, :-1]
train_target=df.iloc[:80, -1]
test_target=df.iloc[80:, -1]

"""
print(df.iloc[1].dtype.kind," coding is tricky")
print(np.shape(train_set),"   train_set")
print(np.shape(test_set),"    test_set")
print(np.shape(train_target), "  train_target")
print(np.shape(test_target),"   test_target")
"""

# way 2 split
"""
msk=np.random.rand(len(df))<.8
train=df[msk]
test=df[msk]
"""
#

# way3 to split
"""
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
"""
#

#Train the model

tree = DecisionTreeClassifier(criterion = 'entropy',splitter='random').fit(train_set,train_target)

# Predict the classes of new, unseen data
test_set=test_set.fillna(test_set.median()) # used to remove the nan values form the test set

prediction = tree.predict(test_set)

#Check the accuracy
test_target=test_target.fillna(test_target.median()) # used to remove the nan values form the test set
print(test_set)
print("potty")
print(test_target)
print("The prediction accuracy is: ",tree.score(test_set,test_target)*100,"%")