import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mmsystem import MM_STREAM_ERROR
from pandas import isnull
import seaborn as sns
from scipy.stats import alpha
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

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
# print(df['Embarked'].unique())
# print(df['Sex'].unique())
df['Embarked']=df['Embarked'].map({'S':1,'C':2,'Q':3})
df['Sex']=df['Sex'].map({'male':1,'female':2})
df['Age']=df['Age'].astype(int)
df['Fare']=df['Fare'].astype(int)

# sns.boxplot(data=df,x='Sex')
# plt.show()
# sns.boxplot(data=df,x='Age')
# plt.show()
#
q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)

IQR=q3-q1

lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR

df=df[(df['Age'] >= lower_bound ) & (df['Age'] <= upper_bound)]


# sns.boxplot(data=df,x='Age')
# plt.show()

# sns.boxplot(data=df,x='Fare')
# plt.show()
# Q1 = df['Fare'].quantile(0.10)
# Q3 = df['Fare'].quantile(0.60)
#
# IQR = Q3- Q1
#
# lower_bound2 = Q1 - 1.5 * IQR
# upper_bound2 = Q3 + 1.5 * IQR
#
# df= df[(df['Fare'] >= lower_bound2) & (df['Fare']<= upper_bound2)]
#
# sns.boxplot(data=df,x='Fare')
# plt.show()


# sns.boxplot(data=df,x='SibSp')
# plt.show()
Q1 = df['SibSp'].quantile(0.10)
Q3 = df['SibSp'].quantile(0.60)

IQR = Q3- Q1

lower_bound2 = Q1 - 1.9 * IQR
upper_bound2 = Q3 + 1.9 * IQR

df= df[(df['SibSp'] >= lower_bound2) & (df['SibSp']<= upper_bound2)]

# sns.boxplot(data=df,x='Fare')
# plt.show()




# print(df.isnull().sum())
# print(df)
# print(df.corr(numeric_only=True))
# print(df.dtypes)
# print(df['Age'].skew())
# sns.histplot(df,x='Age')
# plt.show()
# sns.scatterplot(df,x='Age',y='Fare',hue='Sex',alpha=0.6)
# sns.pairplot(df)
# plt.show()
# print(df.corr())
# sns.heatmap(df.corr())
# plt.show()

X=df.drop('Survived',axis=1)
Y=df['Survived']

ss = StandardScaler()
x_resxale= ss.fit_transform(X)
X_train ,X_test, Y_train ,Y_test =train_test_split(X,Y, test_size=0.01 , random_state=0)


model=LinearRegression()
model.fit(X_train,Y_train)
y_pred= model.predict(X_test)

# model=LogisticRegression()
# model.fit(X_train,Y_train)
# y_pred= model.predict(X_test)

# print(y_pred)

mae='Mae :', mean_absolute_error(Y_test,y_pred)
mse='Mse :', mean_squared_error(Y_test,y_pred)
Rmse ='Rmse :',np.sqrt(mean_squared_error(Y_test,y_pred))
r2_score=r2_score(Y_test,y_pred)

print(mae)
print(mse)
print(Rmse)
print(r2_score)


newdata=[[50,2,1,23,1,0,8,1]]
final_model=LinearRegression()
final_model.fit(X.values,Y)
print(final_model.predict(newdata))




from joblib import dump,load

dump(y_pred ,'Titanic_app')
load_model=load('Titanic_app')



