import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


pd.options.display.max_columns = 2
pd.options.display.max_rows = 100000
data1 = pd.read_csv('csv files/collegereview2021.csv')
data2 = pd.read_csv('csv files/collegereview2022.csv')
data3 = pd.read_csv('csv files/collegereview2023.csv')

def func1(var):
    if any(x.isalpha() or x.isspace() for x in var):
        return -1
    else:return int(var)
data2['Unnamed: 0'] = data2['Unnamed: 0'].apply(func1)
data2.drop_duplicates(inplace=True,keep=False)
data2.reset_index(inplace=True)
data2.drop_duplicates(inplace=True,keep=False)
data2.reset_index(inplace=True)
data2.dropna(axis=0,inplace=True)

def func2(var):
   return int(var)

data3.dropna(axis=0,inplace=True)
data3.reset_index(inplace=True)
data3['Unnamed: 0'] = data3['Unnamed: 0'].apply(func2)


data2.drop(['Name','college','review'],inplace=True,axis=1)
data1.drop(['Name','college','review'],inplace=True,axis=1)
data3.drop(['Name','college','review'],inplace=True,axis=1)


data2['Unnamed: 0'].__iadd__(len(data1)+1)
data3['Unnamed: 0'].__iadd__(len(data1)+1+len(data2)+1)

full_data = pd.concat([data1,data2,data3],join='inner')

x = full_data['Unnamed: 0'].values.reshape(-1,1)
y = full_data['rating'].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
y_train = np.round(y_train)
y_test = np.round(y_test)
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
sns.boxplot(full_data,x='Unnamed: 0',y='rating')
plt.plot(x_test,y_pred)
plt.show()












