# Data Preprocessing

import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder


# Import the dataset
data = pd.read_csv('Data.csv')
X = data.iloc[:, :-1].values    # X is a Numpy Array
y = data.iloc[:, -1].values     # y is a Numpy Array

# Taking care of missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
