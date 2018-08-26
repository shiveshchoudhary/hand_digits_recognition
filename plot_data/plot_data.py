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

d=test[564]
d.shape=(28,28)

pt.imshow(255-d,cmap="gray")
