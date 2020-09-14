import pandas as pd
import os
from sklearn import tree
from sklearn import model_selection

os.chdir("C:/Venkat/Personal/Trainings/Datasets/")

#print(sklearn.__version__)
#creation of data frames from csv
titanic_train = pd.read_csv("Titanic_train.csv")

titanic_train.info()
titanic_train.shape

#convert categorical features to one-hot encoded continuous features
features = ['Pclass', 'Sex', 'Embarked']
titanic_train1 = pd.get_dummies(titanic_train, columns=features)
print(titanic_train1.shape)

#Drop not useful features  for learning pattern
features_to_drop = ['PassengerId', 'Survived', 'Name', 'Age', 'Ticket', 'Cabin']
titanic_train1.drop(features_to_drop, axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train[['Survived']]

# Overfitting (max possible depth of te tree)
classifier = tree.DecisionTreeClassifier()
#learn the pattern automatically
classifier.fit(X_train, y_train)
# Validate and Calculate accuracy score on total data
classifier.score(X_train, y_train)
# Validate and Calculate accuracy score on K-fold(Cross Validate)
scores = model_selection.cross_validate(classifier, X_train, y_train, cv = 10,return_train_score=True)
print(scores)
print(scores.get('test_score').mean())
print(scores.get('train_score').mean())

#prunning the tree depth
# Under fitting
dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'max_depth':list(range(1,4))}
grid_dt_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10)
grid_dt_estimator.fit(X_train, y_train)

print(grid_dt_estimator.best_estimator_)
print(grid_dt_estimator.best_params_)
#find the cv and train scores of final model
print(grid_dt_estimator.best_score_)
print(grid_dt_estimator.score(X_train, y_train))

#Right fit
dt_estimator = tree.DecisionTreeClassifier(random_state=100)
dt_grid = {'max_depth':list(range(3,12))}
grid_dt_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10)
grid_dt_estimator.fit(X_train, y_train)

print(grid_dt_estimator.best_estimator_)
print(grid_dt_estimator.best_params_)
#find the cv and train scores of final model
print(grid_dt_estimator.best_score_)
print(grid_dt_estimator.score(X_train, y_train))

