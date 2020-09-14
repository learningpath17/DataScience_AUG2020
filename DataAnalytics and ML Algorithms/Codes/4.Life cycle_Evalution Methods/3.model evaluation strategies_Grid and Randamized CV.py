import pandas as pd
from sklearn import tree
import pydot
import io
import os
from sklearn import preprocessing, model_selection
from sklearn.impute import SimpleImputer
#from sklearn_pandas import CategoricalImputer

os.chdir("C:/Venkat/Personal/Trainings/Datasets/")

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

features = ['Pclass', 'Parch' , 'SibSp', 'Age', 'Fare', 'Embarked', 'Sex']
X_train = titanic_train[features]
y_train = titanic_train['Survived']

#create an instance of decision tree classifier type
dt_estimator = tree.DecisionTreeClassifier()

#grid search 
dt_grid = {'criterion':["gini", "entropy"], 'max_depth':[3,4,5,6,7], 'min_samples_split':[2,10,20,30]}
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10,return_train_score=True)
dt_grid_estimator.fit(X_train, y_train)

#access the results
print(dt_grid_estimator.best_params_)
print(dt_grid_estimator.best_score_)
final_estimator = dt_grid_estimator.best_estimator_
results = dt_grid_estimator.cv_results_
print(results.get("mean_test_score"))
print(results.get("mean_train_score"))
print(results.get("params"))

#get the logic or model learned by Algorithm
#issue: not readable
print(final_estimator.tree_)

#get the readable tree structure from tree_ object
#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(final_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")
graph.write_pdf("tree_GridsearchCV.pdf")


#Random search
dt_rand_estimator = model_selection.RandomizedSearchCV(dt_estimator, dt_grid, cv=10, n_iter=20)
dt_rand_estimator.fit(X_train, y_train)

#access the results
print(dt_rand_estimator.best_params_)
print(dt_rand_estimator.best_score_)
final_estimator_rand = dt_rand_estimator.best_estimator_
results = dt_rand_estimator.cv_results_
print(results.get("mean_test_score"))
print(results.get("mean_train_score"))
print(results.get("params"))

#get the logic or model learned by Algorithm
#issue: not readable
print(final_estimator_rand.tree_)

#get the readable tree structure from tree_ object
#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(final_estimator_rand, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("tree_RandomizedsearchCV.pdf")

#Read test data
os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets")

titanic_test = pd.read_csv("Titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])

X_test = titanic_test[features]
os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")

#based on Gridsearch CV
titanic_test['Survived'] = final_estimator.predict(X_test)
titanic_test.to_csv("submission_GridSeacrhCV.csv", columns=["PassengerId", "Survived"], index=False)

#based on Gridsearch CV
titanic_test['Survived'] = final_estimator_rand.predict(X_test)
titanic_test.to_csv("submission_GridSeacrhCV.csv", columns=["PassengerId", "Survived"], index=False)



