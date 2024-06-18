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

X=Norm_df.drop('quality',axis=1)
y=Norm_df['quality'].apply(lambda y_value: 1 if y_value>=0.8 else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Model=Sequential(
     [Dense(units=10,activation='relu',name='layer1'),
      Dense(units=8,activation='relu',name='layer2')
      ,Dense(units=1,activation='sigmoid',name='layer3')])

Model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.001),metrics=['accuracy'])
Model.fit(X_train,y_train,epochs=100)


Model.summary()
print(Model.evaluate(X_test,y_test))
#RandomForest
Model2=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
Model2.fit(X_train,y_train)
y_pred2=Model2.predict(X_test)
y_pred=Model2.predict(X_train)

print(accuracy_score(y_train,y_pred))
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_train,y_pred))
