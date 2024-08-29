import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame , concat
from sklearn.metrics import mean_absolute_error , mean_squared_error
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation
from sklearn.preprocessing import LabelEncoder
from numpy import array , hstack
from tensorflow import keras
import tensorflow as tf
#=============================================================
dataset = pd.read_csv("/Users/amiralibozorgnia/Library/CloudStorage/GoogleDrive-amiralibozorgnia79@gmail.com/My Drive/my computer/Python/data set/LSTM-Multivariate_pollution.csv", header=0, index_col=0)
print(dataset.head())
t = dataset.columns.tolist()
dataset = dataset[['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain','pollution']]
print(t)
print(dataset.head())
print(dataset.info())
#=============================================================
#EDA of data
print(dataset.isnull().sum())
encoder = LabelEncoder()
encoder.fit(dataset['wnd_dir'].values)
dataset['wnd_dir'] = encoder.transform(dataset['wnd_dir'].values)
print(dataset.head())
#=============================================================
#Create_sequence_data
def seq_data(window_size, data,target):
    X=[]
    Y=[]
    for i in range(len(data)-window_size):
        window = data.iloc[i:i+window_size].values
        label= data.iloc[i+window_size][target]
        X.append(window)
        Y.append(label)
    return np.array(X), np.array(Y)
#=============================================================
#Visualization
import matplotlib.cm as cm
values=dataset.values
groups = [0,1, 2, 3,4,5,6,7]
i = 1
# plot each column
fig, axs = plt.subplots(len(groups), 1, figsize=(20, 14), facecolor='white')
for group, ax in zip(groups, axs):
    ax.plot(values[:, group], color=cm.viridis(group/len(groups))) #color map just accept range between 0 to 1
    ax.set_title(dataset.columns[group], y=0.75, loc='right', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    i += 1

plt.suptitle('Time Series Plot of Selected Variables', fontsize=24)
plt.tight_layout()
plt.show()
v=zip(groups,axs)
print(list(v))

import seaborn as sns
plt.figure(figsize=(20, 14))
heat_map=sns.heatmap(data=dataset.corr(), annot=True, fmt='.2f',vmin=0, vmax=1,color='red')
plt.show()
#=============================================================
#normalization
columns = (['pollution', 'dew', 'temp', 'press', 'wnd_spd',
       'snow', 'rain' , "wnd_dir"])

scaler=MinMaxScaler()
# Scale the selected columns to the range 0-1
dataset[columns] = scaler.fit_transform(dataset[columns])
dataset[columns] = scaler.transform(dataset[columns])
print(dataset.head())

X,Y=Train_set_sequence=seq_data(10,dataset,'pollution')
X_train,y_train=X[:35000],Y[:35000]
X_val, y_val = X[35000:], Y[35000:65000]
print(X_train.shape)
#=============================================================
#model_training
