import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score,mean_squared_error,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

# Importing_data_set
df = pd.read_csv(
    '/Users/amiralibozorgnia/Library/CloudStorage/GoogleDrive-amiralibozorgnia79@gmail.com/My Drive/my computer/Python/data set/loan-data.csv')
print(df.head())
print(df.describe())
print(df.info())

# cleaning
print(df.isnull().sum())
df = df.dropna()
df.loc[:, ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']] = df[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']].fillna(df[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']].mean())
categorical_features = df.select_dtypes(include=['object']).columns
encoded_data = pd.get_dummies(df, columns=categorical_features)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

print(df.isnull().sum())
Scalar=MinMaxScaler()
Normalized_df = pd.DataFrame(Scalar.fit_transform(df), columns=df.columns)

#Visualization
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt = '.2f', annot_kws = {'fontsize': 7})
plt.tight_layout()
plt.show()

#Training_Algorithm:
X=Normalized_df.drop(columns=['Loan_Status','Loan_ID'], axis=1)
Y=Normalized_df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2,stratify=Y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


for i in [2,4,6,8,10]:
   Model = RandomForestClassifier(n_estimators=500,oob_score=True,max_leaf_nodes=20,max_depth=i)
   Model.fit(X_train, y_train)
   y_pred = Model.predict(X_test)
   y_train_pred = Model.predict(X_train)
   print("the max leaf node of the tree is :",i)
   print("test set: ", accuracy_score(y_test, y_pred))
   print("test set: ",confusion_matrix(y_test, y_pred))
   print("train set: ",confusion_matrix(y_train_pred,y_train))
   print("test set: ",f1_score(y_test, y_pred))
   print("train set: ",f1_score(y_train_pred, y_train))
   print("train set: ",accuracy_score(y_train_pred,y_train))


# SVCclassifier = svm.SVC(kernel='rbf', max_iter=100)
# SVCclassifier.fit(X_train, y_train)
# y_pred1 = SVCclassifier.predict(X_test)
# print(confusion_matrix(y_test, y_pred1))
# SVCAcc = accuracy_score(y_pred1,y_test)
# print(r2_score(y_test, y_pred1))
# print('SVC accuracy: {:.2f}%'.format(SVCAcc*100))
