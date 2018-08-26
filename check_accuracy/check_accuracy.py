import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("C:/Users/cc102tx/Videos/Captures/train.csv").as_matrix()
cl=DecisionTreeClassifier()
train=data[0:21000,1:]
train_l=data[0:21000,0]

cl.fit(train,train_l)

test=data[21000:,1:]
test_l=data[21000:,0]


p=cl.predict(test)

ct=0
for i in range(21000):
    ct+=1 if p[i]==test_l[i] else 0
print "Accuracy:",(ct*100.0)/21000,"%"


