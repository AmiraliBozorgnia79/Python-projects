import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sns


#Data collection and analysis
data=pd.read_csv('/Users/amiralibozorgnia/Library/CloudStorage/GoogleDrive-amiralibozorgnia79@gmail.com/My Drive/my computer/Python/data set/diabetes.csv')

print(data.head())
print(data.shape)
print(data.describe())

print('the number of outcomes are: ', data['Outcome'].value_counts())
print(data.groupby('Outcome').mean())  #it will group the outcome and analyze them seperately

#Seperating the output and feautures
X=data.drop(columns='Outcome',axis=1)
Y=data['Outcome']

#Data regulization
Scaler=StandardScaler()
Regulized_data=Scaler.fit_transform(X)
X=Regulized_data
Y=data['Outcome']

#Training and test example
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y,random_state=2)

""""In the context of splitting datasets, stratify is
used to ensure that the proportions of different
classes in the dataset are maintained in the training and testing sets.
"""
print(X_train.shape, X_test.shape,  X.shape)

#Training
model=svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

#Evaluation
X_train_pred=model.predict(X_train)
accuracy_train=accuracy_score(X_train_pred,Y_train)
print(accuracy_train)

X_test_pred=model.predict(X_test)
accuracy_test=accuracy_score(X_test_pred,Y_test)
print(accuracy_test)

#Prediction
input=(4,110,92,0,0,37.6,0.191,30)
input_array=np.asarray(input)
input_array2=input_array.reshape(1,-1)
print(input_array)
std_data=Scaler.transform(input_array2)
print(std_data)
prediction=model.predict(std_data)

if prediction==1:
    print("the person has diabetes")
else:
    print("the person does not have diabetes")