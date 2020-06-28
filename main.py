# Author: Kunj Dave
# Data created: 14/05/2020

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier  # ML model
from sklearn.metrics import roc_auc_score

# Read data into a pandas Data frame
data = pd.read_csv('parkinsons.data')

# Clean data
data = data.drop('name',axis=1)  # Remove the participant label column
data = data.drop(['MDVP:Jitter(%)', 'MDVP:Shimmer(dB)', 'spread1', 'Shimmer:APQ3', 'Shimmer:DDA', 'NHR', 'MDVP:RAP',
                  'MDVP:APQ', 'Shimmer:APQ5', 'Shimmer:DDA', 'MDVP:Flo(Hz)', 'MDVP:PPQ', 'MDVP:Fo(Hz)'], axis=1)

# Check if there are any missing values - No missing values
# print(data.isnull().values.any())

# Create features and labels
y = data['status'].values
x = data.drop('status', axis=1).values[:,:]

# Used for testing the has_parkinson() function
test_samples = x[:,:]

# Split data for testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Initiate the ML model to be used
model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=13)
model.fit(x_train, y_train)


def disp_model_stats():
    print('----model metrics----')
    # Calculate the model accuracy
    predicted = model.predict(x_test)
    print('Accuracy: ' + str(accuracy_score(y_test, predicted)*100) + '%')

    # ROC AUC curve (From the MS Learning module)
    prob = model.predict_proba(x_test)
    roc_auc_acc = round(roc_auc_score(y_test, prob[:, 1]) * 100, 2)
    print('ROC_AUC_accuracy: ' + str(roc_auc_acc) + '%')


def has_parkinson(voice_measures):
    # This function takes 10 input variables and returns an array detecting whether the person with the input
    # characteristics is predicted to have Parkinson's Disease or not
    #
    # Input: voice_measures is an array of length 10 with the followinng information in each index in order:
    #        MDVP_Fhi = Maximum vocal fundamental frequency (in Hz)
    #        MDVP_Jitter = The absolute value of jitter in the voice frequency
    #        Jitter_DDP = another measure of variation in fundamental frequency
    #        MDVP_Shimmer = Measure of variation in amplitude
    #        HNR = measure of ratio of noise to tonal components in the person's voice
    #        RPDE = The recurrence period density entropy (nonlinear dynamical complexity measures)
    #        DFA = Signal fractual scaling exponent
    #        spread2 = Nonlinear measure of fundamental frequency variation
    #        D2 = another nonlinear dynamical complexity measure
    #        PPE = Nonlinear measure of fundamental frequency variation
    #
    # Output: An array of length 2 with:
    #         first index = Boolean (True if the person is predicted to have Parkinson's disease)
    #         second index = prediction percentage (The closer to 1 the more confident the model is)

    # Convert input into a numpy array
    input_measures = np.array(voice_measures)
    input_measures = input_measures.reshape(1,-1)

    # Make prediction
    prediction = model.predict_proba(input_measures)
    prediction = round(prediction[0][1])
    if prediction == 1:
        return [True, prediction]
    else:
        return [False, prediction]


# To display model metrics
disp_model_stats()

# For testing the has_parkinson() function - the samples are from the original extracted data set
# print('--------Test---------')
# result = has_parkinson(test_samples[56,:])  # Change the row index to change sample (change on line 94 as well)
# print('Actual output: ' + str(y[56]))
# print('Function output: ' + str(result[0]))
