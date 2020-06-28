# Author: Kunj Dave
# Data created: 14/05/2020

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier  # ML model
from sklearn.metrics import roc_auc_score

# Read data into a pandas Data frame
data = pd.read_csv('parkinsons.data')

# Clean data
data = data.drop('name',axis=1)  # Remove the participant label column
data = data.drop(['MDVP:Jitter(%)', 'MDVP:Shimmer(dB)', 'spread1', 'Shimmer:APQ3', 'Shimmer:DDA', 'NHR', 'MDVP:RAP'], axis=1)

# Check if there are any missing values - No missing values
# print(data.isnull().values.any())

# Create features and labels
y = data['status'].values
x = data.drop('status', axis=1).values[:,1:]

# Check form
# print(x.shape)
# print(y.shape)

# normalise x into range -1 to 1 due to the varying magnitude of the independent variables
# from https://data-flair.training/blogs/python-machine-learning-project-detecting-parkinson-disease/
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(x)

# Split data for testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Initiate the AI model to be used
model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=13)
model.fit(x_train, y_train)

def disp_model_stats():
    # Calculate the model accuracy
    predicted = model.predict(x_test)
    print('Accuracy: ' + str(accuracy_score(y_test, predicted)))

    # ROC AUC curve (From the MS Learning module)
    prob = model.predict_proba(x_test)
    roc_auc_acc = round(roc_auc_score(y_test, prob[:, 1]) * 100, 2)
    print('ROC_AUC_accuracy: ' + str(roc_auc_acc) + '%')

# Function to predict whether a person is healthy or has Parkinson's disease given some voice characteristics
def has_parkinson():
    input = 0
    prediction = model.predict_proba(input)
    if prediction > 0.5:
        return [True, prediction]
    else:
        return [False, prediction]

disp_model_stats()