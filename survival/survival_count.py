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