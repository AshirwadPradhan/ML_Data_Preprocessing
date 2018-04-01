import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer as im
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.cross_validation import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ss

# import the dataset
dataset = pd.read_csv('data\Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# replace missing data in X using mean of the whole column
imputer = im(missing_values='NaN', strategy='mean',
                            axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encode categorical data
labelencode_X = le()
X[:, 0] = labelencode_X.fit_transform(X[:, 0])
# dummy encoding the data
ohotencode = ohe(categorical_features=[0])
X = ohotencode.fit_transform(X).toarray()

labelencode_Y = le()
y = labelencode_Y.fit_transform(y)

# splitting the data into train and test set
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2,
                                       random_state=0)

# feature scaling
standardscale_X = ss()
X_train = standardscale_X.fit_transform(X_train)
X_test = standardscale_X.transform(X_test)
