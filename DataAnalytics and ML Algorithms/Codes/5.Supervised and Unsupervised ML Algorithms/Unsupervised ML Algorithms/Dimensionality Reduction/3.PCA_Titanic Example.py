# importing required libraries
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn import tree  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import impute 
from sklearn import preprocessing, model_selection
from sklearn_pandas import CategoricalImputer

os.chdir('C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets')
# read the train and test dataset
titanic_train = pd.read_csv('Titanic_train.csv')
titanic_test = pd.read_csv('Titanic_test.csv')

# view the top 3 rows of the dataset
print(titanic_train.head(3))
print(titanic_train.info())

# shape of the dataset
print('\nShape of training data :',titanic_train.shape)
print('\nShape of testing data :',titanic_test.shape)

# Now, we need to predict the missing target variable in the test data
# target variable - Survived
#preprocessing stage
#impute missing values for continuous features
imputable_cont_features = ['Age', 'Fare']
cont_imputer = impute.SimpleImputer()
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
X_train = titanic_train[features]
y_train = titanic_train['Survived']
#create an instance of decision tree classifier type
dt_estimator = tree.DecisionTreeClassifier()

#grid search 
dt_grid = {'criterion':["gini", "entropy"], 'max_depth':[3,4,5,6,7], 'min_samples_split':[2,10,20,30]}
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, cv=10)
dt_grid_estimator.fit(X_train, y_train)

#access the results
print(dt_grid_estimator.best_params_)
print(dt_grid_estimator.best_score_)
final_estimator = dt_grid_estimator.best_estimator_
results = dt_grid_estimator.cv_results_
print(results.get("mean_test_score"))
print(results.get("mean_train_score"))
print(results.get("params"))

#read test data
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])

X_test = titanic_test[features]
titanic_test['Survived'] = final_estimator.predict(X_test)

y_test=titanic_test['Survived']

###################

# create object of model
model = LinearRegression()

# fit the model with the training data
model.fit(X_train,y_train)

# predict the target on the train dataset
predict_train = model.predict(X_train)

# Accuray Score on train dataset
rmse_train = mean_squared_error(y_train,predict_train)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)

# predict the target on the test dataset
predict_test = model.predict(X_test)

# Accuracy Score on test dataset
rmse_test = mean_squared_error(y_test,predict_test)**(0.5)
print('\nRMSE on test dataset : ', rmse_test)

# create the object of the PCA (Principal Component Analysis) model
# reduce the dimensions of the data to 12
'''
You can also add other parameters and test your code here
Some parameters are : svd_solver, iterated_power
Documentation of sklearn PCA:

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
'''
model_pca = PCA(n_components=6)

new_train = model_pca.fit_transform(X_train)
new_test  = model_pca.fit_transform(X_test)

print('\nTraining model with {} dimensions.'.format(new_train.shape[1]))

# create object of model
model_new = LinearRegression()

# fit the model with the training data
model_new.fit(new_train,y_train)

# predict the target on the new train dataset
predict_train_pca = model_new.predict(new_train)

# Accuray Score on train dataset
rmse_train_pca = mean_squared_error(y_train,predict_train_pca)**(0.5)
print('\nRMSE on new train dataset : ', rmse_train_pca)

# predict the target on the new test dataset
predict_test_pca = model_new.predict(new_test)

# Accuracy Score on test dataset
rmse_test_pca = mean_squared_error(y_test,predict_test_pca)**(0.5)
print('\nRMSE on new test dataset : ', rmse_test_pca)