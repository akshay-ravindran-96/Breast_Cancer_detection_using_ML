#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import numpy
import matplotlib
import pandas
import sklearn










import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd



#loading the dataset
url ="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names= ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

df = pd.read_csv(url, names= names)


df.replace('?', -9999, inplace= True)
df.drop(['id'], 1, inplace=True)
print(df.axes)




#Print the shape
print(df.axes)
print(df.shape)
print(df.loc[698])




print(df.describe())




df.hist(figsize = (10,10))
plt.show()









scatter_matrix(df, figsize =(18, 18))
plt.show()




# Create x and Y datasets for training

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])





y=np.array(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 )




# Specify testing

seed = 8

scoring = 'accuracy'



models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM', SVC()))


results = []
names =[]

for name, model in models:
    kfold  = model_selection.KFold(n_splits=10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg ="%s:  %f (%f) " % (name, cv_results.mean(), cv_results.std())
    print(msg)



#Make prediction on validation_set


for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))



clf =SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)

example = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 10]])
example = example.reshape(len(example), -1)
prediction = clf.predict(example)

print(prediction)






