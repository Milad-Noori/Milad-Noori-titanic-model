import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import isnull
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("titanic.csv",usecols=['PassengerId','Survived','Pclass'
                                        ,'Name','Sex','Age',
                                        'SibSp','Parch','Ticket','Fare','Cabin','Embarked'])

df=pd.DataFrame(data)

median=df['Age'].median()
df['Age']= df['Age'].fillna(median)
print(df)
print(df.isnull().sum())
