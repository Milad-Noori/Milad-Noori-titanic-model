import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import isnull
import seaborn as sns
from scipy.stats import alpha

data=pd.read_csv("titanic.csv",usecols=['PassengerId','Survived','Pclass'
                                        ,'Name','Sex','Age',
                                        'SibSp','Parch','Ticket','Fare','Cabin','Embarked'])

df=pd.DataFrame(data)
####
# avg_age=df['Age'].mean()
#####
pd.set_option('display.max_row',None)
pd.set_option('display.max_columns',None)

# print(pd.get_option('display.max_row',None))
# print(pd.get_option('display.max_columns',None))

median=df['Age'].median()
df['Age']= df['Age'].fillna(median)

embarked_mode=df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)
df=df.drop('Cabin',axis=1)
df=df.drop('Name',axis=1)
df=df.drop('Ticket',axis=1)
print(df['Embarked'].unique())
print(df['Sex'].unique())
df['Embarked']=df['Embarked'].map({'S':1,'C':2,'Q':3})
df['Sex']=df['Sex'].map({'male':1,'female':2})

# print(df.isnull().sum())
# print(df)
# print(df.corr(numeric_only=True))
print(df.dtypes)
print(df['Age'].skew())
# sns.histplot(df,x='Age')
sns.scatterplot(df,x='Age',y='Fare',hue='Sex',alpha=0.6)
plt.show()