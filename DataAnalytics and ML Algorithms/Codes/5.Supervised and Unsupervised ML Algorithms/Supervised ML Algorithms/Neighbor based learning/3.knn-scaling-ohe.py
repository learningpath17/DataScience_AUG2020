import pandas as pd
import os
from sklearn import neighbors
from sklearn import preprocessing, model_selection
from sklearn.impute import SimpleImputer # inplace of preprocessing Impute we can use SimpleImputer.
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

le_pclass = preprocessing.LabelEncoder()
le_pclass.fit(titanic_train['Pclass'])
print(le_pclass.classes_)
titanic_train['Pclass'] = le_pclass.transform(titanic_train['Pclass'])

categorical_features = ['Pclass', 'Sex', 'Embarked']
ohe = preprocessing.OneHotEncoder()
ohe.fit(titanic_train[categorical_features])
#print(ohe.n_values_)
tmp1 = ohe.transform(titanic_train[categorical_features]).toarray()
tmp1 = pd.DataFrame(tmp1)

continuous_features = ['Fare', 'Age', 'SibSp', 'Parch']
tmp2 = titanic_train[continuous_features]

tmp = pd.concat([tmp1, tmp2], axis=1)

scaler = preprocessing.StandardScaler() # standardizes a feature by subtracting the mean and then scaling to unit variance. Unit variance means dividing all the values by the standard deviation
X_train = scaler.fit_transform(tmp)
#type(X_train)
y_train = titanic_train['Survived']

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':[5,7,8,10,20], 'weights':['uniform','distance']}
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, cv=10, return_train_score=True)
knn_grid_estimator.fit(X_train, y_train)

print(knn_grid_estimator.best_score_)
print(knn_grid_estimator.best_params_)
results = knn_grid_estimator.cv_results_
final_estimator = knn_grid_estimator.best_estimator_
print(final_estimator.score(X_train, y_train))

#read test data
titanic_test = pd.read_csv("Titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])
titanic_test['Pclass'] = le_pclass.transform(titanic_test['Pclass'])
tmp1 = ohe.transform(titanic_test[categorical_features]).toarray()
tmp1 = pd.DataFrame(tmp1)
tmp2 = titanic_test[continuous_features]
tmp = pd.concat([tmp1, tmp2], axis=1)

X_test = scaler.transform(tmp)
titanic_test['Survived'] = final_estimator.predict(X_test)
os.chdir('C:/Venkat/Personal/Trainings/Datasets/Submissions')
titanic_test.to_csv("submission_knn_03.csv", columns=["PassengerId", "Survived"], index=False)
