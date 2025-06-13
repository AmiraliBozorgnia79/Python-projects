import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier



def KNN(X_train, y_train,k,x_query):
    "create a KNN distance list"
    distance_matrix = []
    for i in range(len(X_train)):
    #calculating eucledian distance
        distance = np.sqrt(np.sum((X_train[i] - x_query) ** 2))
        distance_matrix.append(distance)
    #converting distance list to numpy array
    distance_matrix = np.array(distance_matrix)
    #merge training label to x values
    data=np.column_stack((distance_matrix,y_train))
    #sorting distances to find the nearest neighbour
    sort_indices = np.argsort(data[:, 0])
    nearest_neighbors = data[sort_indices[0:k]]
    #design conditions for majority voting process
    label_1=0
    label_0=0
    for j in nearest_neighbors[:,1]:
        if j==0:
            label_0=label_0+1
        else:
            label_1=label_1+1

    if label_0>label_1:
        vote=0
    else:
        vote=1
    #assign the proper label to the query point
    Quary=np.append(x_query,vote)
    print('by considering the number of nigbours',k,'\n the data is: ',Quary, 'the label is :', vote)
    return Quary




"testing the function"
X_train = np.array([[1, 2],
                    [2, 3],
                    [3, 1],
                    [6, 5],
                    [7, 7]])

y_train = np.array([0, 0, 0, 1, 1])

x_query = np.array([5, 5])
#the answer of question 1.3:

KNN(X_train, y_train, 3, x_query)
KNN(X_train, y_train, 5, x_query)











