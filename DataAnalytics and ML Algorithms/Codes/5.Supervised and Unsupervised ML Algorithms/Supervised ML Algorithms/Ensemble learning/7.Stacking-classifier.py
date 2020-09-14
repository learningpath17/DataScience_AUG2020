import pandas as pd
import os
from sklearn import neighbors,tree,linear_model,naive_bayes,ensemble

from sklearn import preprocessing, model_selection
from sklearn_pandas import CategoricalImputer

from sklearn.ensemble import StackingClassifier 

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

le_pclass = preprocessing.LabelEncoder()
le_pclass.fit(titanic_train['Pclass'])
print(le_pclass.classes_)
titanic_train['Pclass'] = le_pclass.transform(titanic_train['Pclass'])

categorical_features = ['Pclass', 'Sex', 'Embarked']
ohe = preprocessing.OneHotEncoder()
ohe.fit(titanic_train[categorical_features])
print(ohe.n_values_)
tmp1 = ohe.transform(titanic_train[categorical_features]).toarray()
tmp1 = pd.DataFrame(tmp1)

continuous_features = ['Fare', 'Age', 'SibSp', 'Parch']
tmp2 = titanic_train[continuous_features]

tmp = pd.concat([tmp1, tmp2], axis=1)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(tmp)
y_train = titanic_train['Survived']

dt_estimator = tree.DecisionTreeClassifier()
dt_grid = {'max_depth':[3,4,5]}
grid_dt_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv= 10)
grid_dt_estimator.fit(X_train, y_train)

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':[3,4,5]}
grid_knn_estimator = model_selection.GridSearchCV(knn_estimator, knn_grid, cv= 10)
grid_knn_estimator.fit(X_train, y_train)

gnb_estimator = naive_bayes.GaussianNB()

rf_estimator = ensemble.RandomForestClassifier()
rf_grid = {'n_estimators':[3,4,5]}
grid_rf_estimator = model_selection.GridSearchCV(rf_estimator, rf_grid, cv= 10)
grid_rf_estimator.fit(X_train, y_train)

lr_estimator = linear_model.LogisticRegression()

from sklearn import classifier  
st_estimator = ensemble.StackingClassifier([dt_estimator,knn_estimator,gnb_estimator,rf_estimator], lr_estimator,store_train_meta_features=True, use_probas=True)
st_grid = {'meta_classifier__C':[0.1,0.2,1]}
grid_st_estimator = model_selection.GridSearchCV(st_estimator, st_grid, cv=10)
grid_st_estimator.fit(X_train, y_train)

print(grid_st_estimator.best_params_)
final_estimator = grid_st_estimator.best_estimator_
print(final_estimator.clfs_)
print(final_estimator.train_meta_features_)
print(grid_st_estimator.best_score_)
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
os.chdir('C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions')
titanic_test.to_csv("submission_Bagging_Stacking_01.csv", columns=["PassengerId", "Survived"], index=False)
