import pandas as pd
import os
from sklearn import neighbors
from sklearn import preprocessing, model_selection
from sklearn_pandas import CategoricalImputer

os.chdir('C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets')
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

features = ['Pclass', 'Parch' , 'SibSp', 'Age', 'Fare', 'Embarked', 'Sex']
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(titanic_train[features])
y_train = titanic_train['Survived']

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':[5,7,8,10,20], 'weights':['uniform','distance']}
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, cv=10, return_train_score=True)
knn_grid_estimator.fit(X_train, y_train)

print(knn_grid_estimator.best_score_)
print(knn_grid_estimator.best_params_)
results = knn_grid_estimator.cv_results_
final_estimator = knn_grid_estimator.best_estimator_

results.get("mean_test_score")
results.get("mean_test_score").mean()

results.get("mean_train_score")
results.get("mean_train_score").mean()
results.get('params')


#read test data
titanic_test = pd.read_csv("Titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])

X_test = scaler.transform(titanic_test[features])
titanic_test['Survived'] = final_estimator.predict(X_test)
os.chdir('C:/Venkat/Personal/Trainings/Datasets/Submissions')
titanic_test.to_csv("submission_knn_02.csv", columns=["PassengerId", "Survived"], index=False)
