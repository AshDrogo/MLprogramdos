#!/usr/bin/python
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#printing our test targets for test data taken from iris example
print test_target

#predicting the labels for given test data where test answer through Ml learning classifier algo
print clf.predict(test_data)

#visualizing the above data to create a pdf
from sklearn.externals.six import StringIO

#alternatively use pydotplus if pydot isn't working
#use conda install pydot or pip install pydot for installing these libraries.
import pydot
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf") 
