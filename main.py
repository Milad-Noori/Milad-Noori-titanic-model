import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import isnull
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("titanic.csv",usecols=['PassengerId','Survived','Pclass'
                                        ,'Name','Sex','Age',
                                        'SibSp','Parch','Ticket','Fare','Cabin','Embarked'])

df=pd.DataFrame(data)
####
# avg_age=df['Age'].mean()
#####
median=df['Age'].median()
df['Age']= df['Age'].fillna(median)

embarked_mode=df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)
df=df.drop('Cabin',axis=1)
print(df.isnull().sum())
# print(df.info())
