import pandas as pd
import os
from sklearn import tree, model_selection

path = 'C:\\Venkat\\Personal\\Trainings\ML\\Datasets'
titanic_train = pd.read_csv(os.path.join(path, 'train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

#features = ['Sex', 'Pclass', 'Embarked','Parch','SibSp']

titanic_train1 = pd.get_dummies(titanic_train, columns=['Sex','Pclass','Embarked'])
X_train = titanic_train1.drop(['PassengerId','Survived','Name','Age','Cabin','Ticket'], axis=1)
y_train = titanic_train[['Survived']]

classifer = tree.DecisionTreeClassifier()
classifer.fit(X_train,y_train)
results = model_selection.cross_validate(classifer, X_train, y_train, cv = 10, return_train_score=True)
print(results.get('test_score').mean())
print(results.get('train_score').mean())

#model implement on test data.
titanic_test = pd.read_csv(os.path.join(path, 'test.csv'))
print(titanic_test.shape)
print(titanic_test.info())
titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Sex','Pclass','Embarked'])
X_test = titanic_test1.drop(['PassengerId','Name','Age','Cabin','Ticket'], axis=1)
titanic_test['Survived'] = classifer.predict(X_test)
titanic_test.to_csv(os.path.join(path,'submission_v4.csv'), columns=['PassengerId','Survived'], index=False)
