import pandas as pd
import os
from sklearn import dummy

os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets")
titanic_train = pd.read_csv('Titanic_train.csv')

print(titanic_train.info())
print(titanic_train.columns)

X_train = titanic_train[ ['SibSp', 'Parch'] ]
y_train = titanic_train['Survived']

# most frequent value filling in all the target rows.
dummy_estimator = dummy.DummyClassifier(strategy="most_frequent")
model=dummy_estimator.fit(X_train, y_train)
model.score(X_train, y_train)

titanic_test = pd.read_csv('Titanic_test.csv')
print(titanic_test.info())
X_test = titanic_test[ ['SibSp', 'Parch'] ]
titanic_test['Survived'] = dummy_estimator.predict(X_test)

# Submission file creation path
os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")

titanic_test.to_csv('submission_04.csv', columns=['PassengerId', 'Survived'], index=False)

# 2. uniform strategy.

dummy_estimator = dummy.DummyClassifier(strategy="uniform", random_state=10)
model=dummy_estimator.fit(X_train, y_train)
model.score(X_train, y_train)

X_test = titanic_test[ ['SibSp', 'Parch'] ]
titanic_test['Survived'] = dummy_estimator.predict(X_test)

titanic_test.to_csv('submission_05.csv', columns=['PassengerId', 'Survived'], index=False)

#3.stratified strategy.
dummy_estimator = dummy.DummyClassifier(strategy="stratified", random_state=10)
model=dummy_estimator.fit(X_train, y_train)
model.score(X_train, y_train)

X_test = titanic_test[ ['SibSp', 'Parch'] ]
titanic_test['Survived'] = dummy_estimator.predict(X_test)

titanic_test.to_csv('submission_06.csv', columns=['PassengerId', 'Survived'], index=False)
