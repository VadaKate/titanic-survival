import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

#Showing impact of sex on survival rates
f, ax = plt.subplots(1, 2, figsize=(12, 4))
train[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0], rot=0)
ax[0].set_title('Survivors by Sex')
sns.countplot(x='Sex', hue='Survived', data=train, ax=ax[1])
ax[1].set_ylabel('Number of People')
ax[1].set_title('Survived (1) & Deceased (0): Men & Women')
plt.show()
