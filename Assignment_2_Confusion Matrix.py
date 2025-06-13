from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay



"loading the dataset by fetch_openml command-based on research version 1 has been used"
mnist = fetch_openml('mnist_784', version=1) # fill this line
print(mnist.data.head())
print(mnist.target.head())

"splitting the dataset into input and targets sets"
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:30000], X[30000:40000], y[:30000], y[30000:40000]

"converting to correct dtypes"
y_train = y_train.astype("int")
y_test = y_test.astype("int")

"training SGD without scaling"
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
print("Unscaled score:", sgd_clf.score(X_test, y_test))


"training SGD with Standard Scaler"
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))  #------ fill this line use astype("float64")
X_test_scaled = scaler.transform(X_test.astype("float64"))
sgd_clf.fit(X_train_scaled, y_train)
print("Standard Scaler score:", sgd_clf.score(X_test_scaled, y_test))

"training SGD with Min-Max Scaler"
#scaler2 = MinMaxScaler()
#X_train_MM = scaler2.fit_transform(X_train.astype("float64"))
#X_test_MM = scaler2.transform(X_test.astype("float64"))
#sgd_clf.fit(X_train_MM, y_train)
#print("Min-Max Scaler score:", sgd_clf.score(X_test_MM, y_test))

"generating cross-validated predictions for confusion matrix"
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

"plotting the confusion matrix"
plt.rc('font', size=9)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, cmap=plt.cm.Greens)
plt.title("Confusion Matrix from Cross-Validated Predictions")
plt.show()

"Normalize the confusion matrix"
plt.rc('font', size=6)
ConfusionMatrixDisplay.from_predictions(y_train,y_train_pred,normalize="true",cmap=plt.cm.Blues)       #----- fill here
plt.show()

"Make the errors more significant and put zero weight on the correct predictions"
sample_weight = (y_train_pred != y_train)
plt.rc('font', size=8)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,sample_weight=sample_weight,normalize="true", values_format=".0%")
plt.show()