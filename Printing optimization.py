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
#==================================================================
#data_structure
data_path='/Users/amiralibozorgnia/Library/CloudStorage/GoogleDrive-amiralibozorgnia79@gmail.com/My Drive/my computer/ML/printing.csv'
data = pd.read_csv(data_path)
print(data.head(),'/')
print(data.info())
print(data.isnull().sum())
#==================================================================
"Label encoding, converting patterns and materials to integers"

categorial_feature=data.select_dtypes(include=['object']).columns
encoded_data = pd.get_dummies(data, columns=categorial_feature)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for feature in categorial_feature:
    data[feature] = label_encoder.fit_transform(data[feature])
print(data.info())
#==================================================================
"data visualization to identify outliers and missing"
for i in data.columns:
    data.boxplot(column=i)
    #plt.show()

"finding effective parameters"
plt.figure(figsize=(14,14))
heat_map=sns.heatmap(data.corr(), annot=True, cmap='YlGnBu',vmin=0,vmax=1)
plt.show()
#==================================================================
