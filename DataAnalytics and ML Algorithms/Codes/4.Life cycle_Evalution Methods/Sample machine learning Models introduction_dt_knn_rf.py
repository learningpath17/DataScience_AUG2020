import pandas as pd
import os
from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble

os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets")

titanic_train = pd.read_csv('Titanic_train.csv')
print(titanic_train.info())
print(titanic_train.columns)

X_train = titanic_train[ ['SibSp', 'Parch'] ]
y_train = titanic_train['Survived']
#knn Model
knn_estimator = neighbors.KNeighborsClassifier()
knn_estimator.fit(X_train, y_train)

# Decission tree
dt_estimator = tree.DecisionTreeClassifier()
dt_estimator.fit(X_train, y_train)

# random forest
rf_estimator = ensemble.RandomForestClassifier()
rf_estimator.fit(X_train, y_train)

# apply all models on test data set
titanic_test = pd.read_csv('Titanic_test.csv')
print(titanic_test.info())
X_test = titanic_test[ ['SibSp', 'Parch'] ]

os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")

#Knn 
titanic_test['Survived'] = knn_estimator.predict(X_test)
titanic_test.to_csv(('submission_knn.csv'), columns=['PassengerId', 'Survived'], index=False)

# Decission tree
titanic_test['Survived'] = dt_estimator.predict(X_test)
titanic_test.to_csv(('submission_dt.csv'), columns=['PassengerId', 'Survived'], index=False)

#Randon forest
titanic_test['Survived'] = rf_estimator.predict(X_test)
titanic_test.to_csv(('submission_rf.csv'), columns=['PassengerId', 'Survived'], index=False)
