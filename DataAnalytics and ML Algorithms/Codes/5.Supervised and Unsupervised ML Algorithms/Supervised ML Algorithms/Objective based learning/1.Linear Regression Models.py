import pandas as pd
import os
from sklearn import preprocessing, model_selection
from sklearn_pandas import CategoricalImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

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

# below linear algorithms are only for continues , included here for understand.
# Linear regression model.
lreg = LinearRegression()
lreg.fit(X_train,y_train)

#Ridge regression
ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(X_train,y_train)

#Lasso regression
lassoReg = Lasso(alpha=0.3, normalize=True)
lassoReg.fit(X_train,y_train)

#Elastic net regression
ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
ENreg.fit(X_train,y_train)

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
titanic_test['Survived'] = lreg.predict(X_test)
os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")
titanic_test.to_csv("submission_lreg_01.csv", columns=["PassengerId", "Survived"], index=False)
