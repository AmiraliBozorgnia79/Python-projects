
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


data=pd.read_csv('/Users/amiralibozorgnia/Downloads/Breast_cancer_data.csv')
print(data.head())
print(data.info())
print(data.describe())

#preprocessing
whole_list=["mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness",'diagnosis']
features=["mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness"]
scaler = MinMaxScaler()
data2=scaler.fit_transform(data[whole_list])
data2=pd.DataFrame(data2,columns=whole_list)
print(data2.head())
print(data2['diagnosis'].value_counts())

#data_spliting
X=data2[features]
y=data2['diagnosis']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#Neural_Network
model=Sequential(
    [Dense(units=5,activation='relu',name='layer1'),
     Dense(units=7,activation='relu',name='layer2'),
     Dense(units=1,activation='sigmoid',name='layer3')
    ]
)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
accuracy_list=[]
epoch_num=[]
for i in range(100,500,50):
    model.fit(X_train, y_train, epochs=i)
    y_pred2 = model.predict(X_test)
    y_pred = model.predict(X_train)
    y_pred_train_binary = (y_pred > 0.5).astype(int)
    y_pred_test_binary = (y_pred2 > 0.5).astype(int)
    accuracy_list.append(accuracy_score(y_train,y_pred_train_binary))
    epoch_num.append(i)

plt.scatter(epoch_num,accuracy_list,label='ML Performance',color='b',linestyle='--')
plt.legend()
plt.show()
# model.fit(X_train,y_train,epochs=200)
# y_pred2=model.predict(X_test)
# y_pred=model.predict(X_train)
# y_pred_train_binary = (y_pred > 0.5).astype(int)
# y_pred_test_binary = (y_pred2> 0.5).astype(int)
# print(accuracy_score(y_train,y_pred_train_binary))
# print(accuracy_score(y_test,y_pred_test_binary))




# plt.figure(figsize=(14,14))
# heatmap=sns.heatmap(data.corr(), annot=True, fmt='.2f',vmin=-1,vmax=1,cbar=True,cmap='YlGnBu',square=True)
# plt.show()

