import pandas as pd
from sklearn import tree
import pydot
import io
from sklearn import preprocessing
from sklearn.impute import SimpleImputer # inplace of preprocessing Impute we can use SimpleImputer.
#from sklearn_pandas import CategoricalImputer
import os
#import sklearn
os.chdir("C:/Venkat/Personal/Trainings/Datasets/")

#print(sklearn.__version__)
#creation of data frames from csv
titanic_train = pd.read_csv("Titanic_train.csv")
print(titanic_train.info())

#preprocessing stage

#The SimpleImputer class is for continues numerical values, but also supports categorical data represented 
#as string values or pandas categoricals when using the 'most_frequent' or 'constant' strategy:
#SimpleImputer(strategy="most_frequent")

#impute missing values for continuous features
imputable_cont_features = ['Age', 'Fare']
#cont_imputer = preprocessing.Imputer() not working in this version python.
cont_imputer = SimpleImputer() # default Mean
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

#impute missing values for categorical features
#cat_imputer = CategoricalImputer
cat_imputer=SimpleImputer(strategy="most_frequent")
cat_imputer.fit(titanic_train[['Embarked']])
#print(cat_imputer.fill_)
print(cat_imputer.statistics_)
titanic_train[['Embarked']] = cat_imputer.transform(titanic_train[['Embarked']])

le_embarked = preprocessing.LabelEncoder()
le_embarked.fit(titanic_train['Embarked'])
print(le_embarked.classes_)
titanic_train['Embarked'] = le_embarked.transform(titanic_train['Embarked'])

le_sex = preprocessing.LabelEncoder()
le_sex.fit(titanic_train['Sex'])
print(le_sex.classes_)
titanic_train['Sex'] = le_sex.transform(titanic_train['Sex'])

features = ['Pclass', 'Parch' , 'SibSp', 'Age', 'Fare', 'Embarked', 'Sex']
X_train = titanic_train[features]
y_train = titanic_train['Survived']
#create an instance of decision tree classifier type
classifer = tree.DecisionTreeClassifier()
#learn the pattern automatically
classifer.fit(X_train, y_train)

#get the logic or model learned by Algorithm
#issue: not readable
print(classifer.tree_)

os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")
#get the readable tree structure from tree_ object
#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(classifer, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("tree_02.pdf")

#read test data
os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets")

titanic_test = pd.read_csv("Titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = cat_imputer.transform(titanic_test['Sex'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])

X_test = titanic_test[features]
titanic_test['Survived'] = classifer.predict(X_test)
os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")

titanic_test.to_csv("submission_tree_02.csv", columns=["PassengerId", "Survived"], index=False)
