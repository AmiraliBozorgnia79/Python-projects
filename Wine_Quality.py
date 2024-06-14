import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score,mean_squared_error,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid,softmax


df=pd.read_csv('/Users/amiralibozorgnia/Library/CloudStorage/GoogleDrive-amiralibozorgnia79@gmail.com/My Drive/my computer/Python/data set/winequality-red.csv')
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.sum)
print(df.info())
print(df['quality'].value_counts())

#Visualization
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f',vmin=0,vmax=1,cbar=True,cmap='YlGnBu',square=True)
plt.show()

# for i,col in enumerate(df.columns):
#     if col!='quality':
#      plt.scatter(df[col],df['quality'],label=col)
#      plt.xlabel(col)
#      plt.ylabel('quality')
#      plt.legend()
#      plt.show()


#Normalization
Scalar=MinMaxScaler()
Norm_df=pd.DataFrame(Scalar.fit_transform(df),columns=df.columns)
print(Norm_df['quality'].value_counts())
#Neural Network
