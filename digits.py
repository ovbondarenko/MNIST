import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model, model_selection, neighbors
import matplotlib
import matplotlib.pyplot as plt
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# print(os.listdir("../input"))

def read_from_csv(file):
    df = pd.read_csv(file)
    return df

train_df = read_from_csv('train.csv')
test_df = read_from_csv('test.csv')
# print(train_df)

trainlabels = train_df['label']
trainimages = train_df.drop('label', axis = 1)

# Split images and labels into test and training sets:
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    trainimages, trainlabels, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
print(knn)

knn_score = knn.score(X_test, y_test)
print(knn_score)