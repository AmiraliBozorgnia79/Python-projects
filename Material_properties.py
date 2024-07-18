import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid,softmax



#Importing_ Data
data_path='/Users/amiralibozorgnia/Downloads/archive (13)'
data=pd.read_csv(data_path+'/data.csv')
print(data.head())

#Analysis
print('data information table',' \n', data.describe())
print(data.isnull().sum())
#--------------------------------------------------------------
#Feature_Engineering
categorical_features = data.select_dtypes(include=['object']).columns
encoded_data = pd.get_dummies(data, columns=categorical_features)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for feature in categorical_features:
    data[feature] = label_encoder.fit_transform(data[feature])

print(data.columns)
#---------------------------------------------------------------------
#Visualization
plt.figure(figsize=(14,14))
heat_map=sns.heatmap(data.corr(), annot=True, cmap='YlGnBu',vmin=0,vmax=1)
plt.show()

fig, ax = plt.subplots(4, 3, figsize=(14, 14))
ax = ax.flatten()
bins = 10
for i in range(len(data.columns)):
    ax[i].hist(data[data.columns[i]], bins=bins)
    ax[i].set_title(data.columns[i])
    ax[i].set_xticklabels(ax[i].get_xticks(), rotation=45)

plt.tight_layout()
plt.show()
#----------------------------------------------------------------
#Data_trainig
scaler = MinMaxScaler()
data2=scaler.fit_transform(data)
data2 = pd.DataFrame(data2,columns=data.columns)
X=data2.drop(columns=['roughness','tension_strenght','elongation'],axis=1)
y=data2['roughness']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=42)
error = []
for i in range (1,20):
    for j in range(1,20):
       Model = RandomForestRegressor(n_estimators=i,max_depth=j,random_state=42)
       Model.fit(X_train, y_train)
       y_pred = Model.predict(X_train)
       y_pred_test = Model.predict(X_test)
       r2=r2_score(y_pred, y_train)
       test_accuracy=r2_score(y_test, y_pred_test)
       error.append(test_accuracy)
print(max(error))
print(error)



#--------------------------------------------------------------------------------------
#Neural_Network
# np.random.seed(10)
# tf.random.set_seed(10)
# Model2=Sequential([Dense(units=10,activation='relu')
#                    ,Dense(units=5,activation='relu')
#                    ,Dense(units=1,activation='sigmoid')
# ])
#
# Model2.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=tf.keras.losses.MeanSquaredError())
# Model2.fit(X_train,y_train,epochs=400)
# #--------------------------------
# Model2.evaluate(X_test,y_test)
# result=Model2.predict(X_test)
# result_train=Model2.predict(X_train)
# print(r2_score(y_train, result_train))
# print(r2_score(y_test,result))

