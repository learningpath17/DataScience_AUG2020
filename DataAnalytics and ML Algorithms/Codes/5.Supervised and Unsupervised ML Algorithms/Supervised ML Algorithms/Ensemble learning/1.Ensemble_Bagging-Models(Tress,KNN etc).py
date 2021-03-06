import pandas as pd
import os
from sklearn import tree,neighbors,ensemble
from sklearn import preprocessing, model_selection
from sklearn_pandas import CategoricalImputer

os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets")
#creation of data frames from csv
titanic_train = pd.read_csv("Titanic_train.csv")
print(titanic_train.info())

#preprocessing stage
#impute missing values for continuous features
imputable_cont_features = ['Age', 'Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])

le_embarked = preprocessing.LabelEncoder()
le_embarked.fit(titanic_train['Embarked'])
print(le_embarked.classes_)
titanic_train['Embarked'] = le_embarked.transform(titanic_train['Embarked'])

le_sex = preprocessing.LabelEncoder()
le_sex.fit(titanic_train['Sex'])
print(le_sex.classes_)
titanic_train['Sex'] = le_sex.transform(titanic_train['Sex'])

le_pclass = preprocessing.LabelEncoder()
le_pclass.fit(titanic_train['Pclass'])
print(le_pclass.classes_)
titanic_train['Pclass'] = le_pclass.transform(titanic_train['Pclass'])

features = ['Pclass', 'Parch' , 'SibSp', 'Age', 'Fare', 'Embarked', 'Sex']
X_train = titanic_train[features]
y_train = titanic_train['Survived']

##bagged ensemble with decision tree
dt_estimator = tree.DecisionTreeClassifier()
bag_estimator = ensemble.BaggingClassifier(base_estimator=dt_estimator)
bag_grid = {'n_estimators':[10, 50, 100, 200], 'base_estimator__max_depth':[3,4,5,6,7] }
bag_grid_estimator = model_selection.GridSearchCV(bag_estimator, bag_grid, cv=10, return_train_score=True)
bag_grid_estimator.fit(X_train, y_train)

print(bag_grid_estimator.best_score_)
print(bag_grid_estimator.best_params_)
final_estimator = bag_grid_estimator.best_estimator_
final_estimator.score(X_train, y_train)
print(final_estimator.estimators_)

##bagged ensemble with knn
knn_estimator = neighbors.KNeighborsClassifier()
bag_estimator = ensemble.BaggingClassifier(base_estimator=knn_estimator)
bag_grid = {'n_estimators':[10, 50, 100, 200], 'base_estimator__n_neighbors':[3,5,10,20] }
bag_grid_estimator = model_selection.GridSearchCV(bag_estimator, bag_grid, cv=10, return_train_score=True)
bag_grid_estimator.fit(X_train, y_train)

print(bag_grid_estimator.best_score_)
print(bag_grid_estimator.best_params_)
final_estimator = bag_grid_estimator.best_estimator_
final_estimator.score(X_train, y_train)
print(final_estimator.estimators_)

#read test data
titanic_test = pd.read_csv("Titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])

X_test = titanic_test[features]
titanic_test['Survived'] = final_estimator.predict(X_test)
os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")
titanic_test.to_csv("submission_ensemble_01.csv", columns=["PassengerId", "Survived"], index=False)
