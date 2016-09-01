#!/usr/bin/python
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

#The three data entries that will be removed ...
test_idx = [0,50,100]

#training data
#using numpy to delete certain data index entries at test_idx ..
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
#also a clean way of using numpy to extract certain index elements out for us.
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#we use tree.Decision..() to invoke a decision tree classifier
clf = tree.DecisionTreeClassifier()
#we put in the ml learning algo and train the invoked clf classifier in above step using fit()
clf.fit(train_data, train_target)

#printing our test targets for test data taken from iris example
#test_target being iris.target[[0,50,100]]
print test_target

#predicting the labels for given test data where test answer through Ml learning classifier algo
print clf.predict(test_data)

#commenting out the below for a bug free non pdf code xP
"""
#visualizing the above data to create a pdf
from sklearn.externals.six import StringIO  
import pydot
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf") 
"""
print test_data[0]
print test_target[0]

#we can print the feature and targets by looking at the metadata
print iris.feature_names, iris.target_names

#To understand the classifier flow of decision's cheout the screenshot.
