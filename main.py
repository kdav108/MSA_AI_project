# Author: Kunj Dave
# Data created: 14/05/2020

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Read data into a pandas Data frame
data = pd.read_csv('parkinsons.data')

# Get data shape - (195, 24)
# print(data.shape)

# Check if there are any missing values - No missing values
# print(data.isnull().values.any())


# Create features and labels
y = data['status'].values
x = data.drop('status', axis=1).values[:,1:]

# Check form
print(x.shape)
print(y.shape)

# normalise x into range -1 to 1 due tot he varying magnitude
# from https://data-flair.training/blogs/python-machine-learning-project-detecting-parkinson-disease/
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(x)
# print(x)

# Split data for testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)




