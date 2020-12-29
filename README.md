### Hand_digits_recognition
This project is based on recognising hand written digits using machine learning linear classifiers.

### Knowledge required

Numpy
Scikit  
Pandas  
Decision Tree Classifiers  

### TESTING AND TRAINING SETS
[Kaggle testing and training csv file links](https://www.kaggle.com/c/digit-recognizer/data)


### Decision Tree Classifiers
Decision Tree Classifier, repetitively divides the working area(plot) into sub part by identifying lines.

#### Aspects 
#### 1)IMPURITY 
Impurity is when we have a traces of one class division into other. This can arise due to following reason

We run out of available features to divide the class upon.
We tolerate some percentage of impurity (we stop further division) for faster performance. 

#### 2)Entropy 
Entropy is degree of randomness of elements or in other words it is measure of impurity.

#### 3)Information Gain 
Information Gain (n) =Entropy(x) — ([weighted average] * entropy(children for feature)).

### Final Thoughts 
Dividing efficiently based on maximum information gain is key to decision tree classifier. However, in real world with millions of data dividing into pure class in practically not feasible (it may take longer training time) and so we stop at points in nodes of tree when fulfilled with certain parameters (for example impurity percentage).
