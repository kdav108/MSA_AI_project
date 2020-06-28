# Parkinnson's Detector

### Project idea 
Having a passion for Software Engineering in the medical field, this project focuses on using Machine Learning to train a model to differentiate a healthy person from a person with Parkinson's Disease using several biomedical voice measurements of a given person.

### Project background 
This project is part of the submission for the Artificial intelligence project for Microsoft Student Accelerator program (Australia)

### Code Dependencies and Technologies used
Language used:
- Python (version 3.7)

To run the code file successfully, ensure that the following libraries are installed:
- Pandas
- Scikit-learn

### Data Source
The data was downlaoded from https://archive.ics.uci.edu/ml/datasets/parkinsons. There are 195 rows and 23 columns in the raw data set. The data is also slightly biased in that 23 of the 31 participating people tested for their vocal measurements had Parkinnson's Disease. There was the option of removing some of the data sets relating to people with parkinson's Disease, to lessen this biasedness, however, this would reduce the already small data set, so this option was not exercised. 

### How the model was trained
Since the algorithm is about predicting 2 possible outcomes, True or False (i.e. it's a classification problem) and the data set is quite small (only 195 rows), I decided to use Gradient boosting to build my model. Not all variables from the raw data set were used in the final model (selection was based on omitting insignificant variables and repetetive testing) 
The final model was chosen based on it's accuracy and ROC_AUC value.

### How to test the model
- Ensure the correct environment is set-up (see 'Code Dependencies and Technologies used' section)
- The code includes a function called has_parkinson() with function definition included to describe the input values taken
- Run the function to use the model
- To print model statistics, use the function disp_model_stats()

### Model statistics
The following metrics were recorded for the final model:
- Accuracy: 88%
- ROC_AUC: 96%
- Confusion matrix:

### How to test the model?

References: The background to understand the relation between vocal measurements and Parkinson's disease was gained from the following research paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549905/

Author: Kunj Dave
