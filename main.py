import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import isnull

data=pd.read_csv("titanic.csv")
df=pd.DataFrame(data)
print(df.isnull().sum())
print(df.dtypes)

