import pandas as pd
import os
os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets")

titanic_train = pd.read_csv("Titanic_train.csv")
print(titanic_train.shape)

print(titanic_train.info())

#discover pattern: which class is majority?
titanic_train.groupby('Survived').size()

titanic_test = pd.read_csv("Titanic_test.csv")
print(titanic_test.shape)

os.chdir("C:\\Users\\vesuraju\\OneDrive - DXC Production\\Venkat\\Personal\\Trainings\\ML\\Classes_Year 2020\\Codes_2020\\Datasets\\Submissions")

#1.  Assume all are not survived as predeiction  
titanic_test['Survived'] = 0
titanic_test.to_csv("submission_01.csv", columns = ['PassengerId', 'Survived'], index=False)

#2. discover pattern: which class is majority?
titanic_train.groupby('Survived').size()
titanic_train.groupby(['Sex','Survived']).size()

print(titanic_test.shape)

#guess based on patterns, here female has more chnaces to giving 100% as guess
titanic_test['Survived'] = 0
titanic_test.loc[titanic_test.Sex =='female', 'Survived'] = 1
titanic_test.to_csv("submission_02.csv", columns = ['PassengerId', 'Survived'], index=False)

# try to find more patterns with more no of features.

#discover pattern: which class is majority?
titanic_train.groupby('Survived').size()
titanic_train.groupby(['Sex','Survived']).size()
titanic_train.groupby(['Sex','Pclass','Survived']).size()
titanic_train.groupby(['Sex','Embarked','Survived']).size()

titanic_test.loc[(titanic_train.Sex=='female') & (titanic_train.Pclass==1), 'Survived'] = 1 
titanic_test.to_csv("submission_03.csv", columns = ['PassengerId', 'Survived'], index=False)                