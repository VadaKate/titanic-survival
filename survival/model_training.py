import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')

#reading csv data from pandas
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
# To know number of columns and rows
train.shape
# (891, 12)

#Printing data
train.info()

#Checking NULL values
print(train.isnull().sum())

#Optimising data
#Drop Cabin information
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)

#Drop Ticket information
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1) 

#Replace NULL values on Embarked
train = train.fillna({"Embarked": "S"})

#Sort age into set categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager',
          'Student', 'Young Adult', 'Adult',
          'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins,
                           labels=labels)
test['AgeGroup'] = pd.cut(test["Age"], bins,
                          labels=labels)

#Categorising titles
#Create a combined group of both datasets
combine = [train, test]

"""Extract a title for each name in the
train and test datasets"""
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('' \
    '([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major',
         'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(
        ['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'],
                                     as_index=False).mean()

#map each title group into a numerical value
title_mapping = {"Mr": 1, "Miss":2, "Mrs": 3, "Master":4,
                 "Royal":5, "Rare":6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(
        title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#use title information to fill in missing age values
#Young Adult
mr_age = train[train["Title"] == 1]["AgeGroup"].mode()
#Student
miss_age = train[train["Title"] == 2]["AgeGroup"].mode()
#Adult
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode()
#Baby
master_age = train[train["Title"] == 4]["AgeGroup"].mode()
#Adult
royal_age = train[train["Title"] == 5]["AgeGroup"].mode()
#Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode()

age_title_mapping = {1: "Young Adult", 2: "Student",
                     3: "Adult", 4: "Baby", 5:"Adult",
                     6: "Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]


#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,
               'Student': 4, 'Young Adult': 5, 'Adult': 6,
               'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head

#dropping the age feature for now
train = train.drop(['Name', 'Age'], axis=1)
test = test.drop(['Name', 'Age'], axis=1)

#assign numerical values to sex and embark categories
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

#fill in missing fare values with mean
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        #Pclass = 3
        pclass = test["Pclass"][x]
        test["Fare"][x] = round(
            train[train["Pclass"]  == pclass]["Fare"].mean(), 4)
        
"""map Fare values into numerical 
values"""
train['FareBand'] = pd.qcut(train['Fare'], 4,
                            labels=[1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4,
                           labels= [1, 2, 3, 4])

#drop fare values
train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)


#building the predictive model
#data splitting
from sklearn.model_selection import train_test_split

"""Drop the Survived and PassengerId
column from the trainset"""

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(
    predictors, target, test_size=0.2, random_state=0)

#import the random forest function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

randomforest = RandomForestClassifier()

#Fit the training data along with its output
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)

#Find the accuracy score of the model
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)

#generating survival predictions on test data
#running predictions
ids = test['PassengerId']
predictions = randomforest.predict(test.drop('PassengerId', axis=1))

"""set the output as a dataframe and convert
to csv file named resultfile.csv"""
output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
output.to_csv('resultfile.csv', index=False)
