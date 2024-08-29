import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
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
#===========================================================================
#Data_uploading
df=pd.read_csv('/Users/amiralibozorgnia/Downloads/pprxj2yfby-1/Environmental_sensor_dataset.csv')
print(df.head(10))
print(df.isnull().sum())
#===========================================================================
'''Data_Visualization / from datetime import datetime
''Here each feature have been plotted vs time 
to see the variation of each of them'''

feature_list = ['outtemp','Humidity','pressure','Gas']
Color_list = ['red','blue','green','orange']
fig, ax = plt.subplots(2, 2, figsize=(14, 14))
ax = ax.flatten()

for idx, feature in enumerate(feature_list):
    y = np.array(df.loc[:30000, [feature]])
    X = np.arange(len(y))
    ax[idx].plot(X, y, 'o', label=f'{feature} vs Time', color=Color_list[idx])
    ax[idx].set_xlabel('Time')
    ax[idx].set_ylabel(feature)
    ax[idx].set_title(f'{feature} vs Time')
    ax[idx].legend()

plt.tight_layout()
plt.show()
#===========================================================================
'''label encoding for each feature, by using a dictionary and a heatmap function'''

Output_dictionary={'Defect_01':1,'Defect_02':2, 'Defect_03': 3,'Normal':0}
df['activity_name'] = df['activity_name'].map(Output_dictionary)
print(df.head(10))
#===========================================================================
"""now we have a data set with four classes, to obtain a normal condition, 
we have to asses this through different algorithm"""
y=np.array(df.loc[:20000, ['Humidity']])
X=np.arange(len(y))
print(X)
plt.plot(X, y, 'o')
plt.show()
