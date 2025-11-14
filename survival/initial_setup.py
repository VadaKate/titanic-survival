import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')
#%matplotlib inline
warnings.filterwarnings('ignore')

#reading csv data from pandas
train = pd.read_csv(r"C:\py_envs\survival\train.csv")
test = pd.read_csv(r"C:\py_envs\survival\test.csv")

# To know number of columns and rows
train.shape
# (891, 12)

#Printing data
train.info()

#Checking NULL values
print(train.isnull().sum())



#Visualise the number of survivors and death counts
f, ax = plt.subplots(1, 2, figsize=(12, 4))
train['Survived'].value_counts().plot.pie(
    explode=[0, 0.1], autopct='%1.1f%%',
    ax=ax[0], shadow=True)
ax[0].set_title('Survivors (1) & the Dead (0)')
ax[0].set_ylabel('')
sns.countplot(x='Survived', data=train, ax=ax[1])
ax[1].set_ylabel('Number of People')
ax[1].set_title('Survivors (1) & the Dead (0)')
plt.show()

#Showing impact of sex on survival rates
f, ax = plt.subplots(1, 2, figsize=(12, 4))
train[['Sex','Survived']].groupby(
    ['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survivors by Sex')
sns.countplot(x='Sex', hue='Survived',
              data=train, ax=ax[1])
ax[1].set_ylabel('Number of People')
ax[1].set_title('Survived (1) & Deceased (0):' \
'Men & Women')
plt.show()



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