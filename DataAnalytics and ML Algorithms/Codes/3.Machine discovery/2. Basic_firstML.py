import pandas as pd
from sklearn import tree
import pydot
import io
import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#Install GraphViz with conda install graphviz.
os.chdir("C:/Venkat/Personal/Trainings/Datasets/")
#creation of data frames from csv
titanic_train = pd.read_csv('Titanic_train.csv')
print(titanic_train.info())

#want to build model on only selected features.
features = ['Pclass', 'Parch' , 'SibSp']
X_train = titanic_train[features]
y_train = titanic_train['Survived']

#create an instance of decision tree classifier type
classifer = tree.DecisionTreeClassifier()
print(type(classifer))

#learn the pattern automatically
classifer.fit(X_train, y_train)
#classifer.score(X_train, y_train)

#get the logic or model learned by Algorithm
#issue: not readable
print(classifer.tree_)

#os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")
#get the readable tree structure from tree_ object
#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(classifer, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("tree_01.pdf")

#read test data
titanic_test = pd.read_csv("Titanic_test.csv")
print(titanic_test.info())
X_test = titanic_test[features]
titanic_test['Survived'] = classifer.predict(X_test)
#os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")
titanic_test.to_csv("submission_tree_01.csv", columns=["PassengerId", "Survived"], index=False)
