# KNN algorithm for Iris Dataset classification.

# Import necessary modules 
import numpy as np
import pandas as pd
#import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Loading data 
irisData = load_iris()
type(irisData)

#just to analyse the data in iris data set
iris_data = pd.DataFrame(data=irisData.data,columns=irisData.feature_names,dtype=np.float32)

iris_data.shape
iris_data.info()
iris_data.head()

# Create feature and target arrays 
X = irisData.data 
y = irisData.target 

# Split into training and test set 
X_train, X_test, y_train, y_test = train_test_split( 
             X, y, test_size = 0.2, random_state=42) 
  
knn = KNeighborsClassifier(n_neighbors=4) 
knn.fit(X_train, y_train) 

#y_pred is prediction on X_train data samples by using knn  model (actual is y_train)
y_train_pred = knn.predict(X_train)

#Accuracy Score
print(knn.score(X_train, y_train))

#Accuracy Score, Classification report & Confusion Matrix on train
print(accuracy_score(y_train, y_train_pred))  # equal to knn.score(X_train, y_train)
print(classification_report(y_train, y_train_pred))
print(confusion_matrix(y_train, y_train_pred))
  
# Predict on dataset which model has not seen before 
print(knn.predict(X_test)) 
#y_pred is prediction on Xtest data samples for y_test
y_test_pred = knn.predict(X_test)

print(knn.score(X_test, y_test)) 

#Accuracy Score, Classification report & Confusion Matrix on test
print(accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))


#to get the range of expected k-value, but to get the exact k-value we need to test the model for each and every expected k-value.

neighbors = np.arange(1, 9) 
train_accuracy = np.empty(len(neighbors)) 
test_accuracy = np.empty(len(neighbors)) 
  
# Loop over K values 
for i, k in enumerate(neighbors): 
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train) 
      
    # Compute traning and test data accuracy 
    train_accuracy[i] = knn.score(X_train, y_train) 
    test_accuracy[i] = knn.score(X_test, y_test) 
  
# Generate plot 
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 
  
plt.legend() 
plt.xlabel('n_neighbors') 
plt.ylabel('Accuracy') 
plt.show() 


