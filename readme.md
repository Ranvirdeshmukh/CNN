
Overview
This project implements a neural network for predicting parts of speech (POS) in Cook Islands Māori. The code is based on a tutorial from Machine Learning Mastery, and it has been adapted for our linguistic dataset. The goal is to classify tokens into one of three parts of speech: noun, verb, or preposition.

Project Structure
Data Preparation:

The CSV file cim-3pos.csv is loaded using pandas and then preprocessed.
Features (predictive words) are one-hot encoded, and the target variable (POS tags) is label-encoded and transformed into a categorical format.
The dataset is split into training (90%) and test (10%) sets.
Neural Network Model:

The model is built using Keras with TensorFlow as the backend.
Original configuration:
A hidden layer with 48 neurons.
A second hidden layer with 24 neurons (changed from an earlier version which had 6 neurons).
An output layer with 3 neurons using the softmax activation function.
The model is compiled with the categorical_crossentropy loss function and the adam optimizer.
The model is then trained for a specified number of epochs (default is 200 in the modified code).
Evaluation:

After training, the model makes predictions on the test set.
The code prints:
The raw probabilities for the first five test items.
The first 15 test inputs, along with predicted and true labels.
Overall test accuracy, a classification report, and a confusion matrix.
These outputs help in evaluating model performance, including the F1-scores.
Homework Tasks
The project includes three distinct tasks that require multiple runs and modifications:

Task (a): Baseline Experiment
Objective:
Run the program three times.
Record the training and test accuracy for each run.
Analyze how the F1-scores behave by reviewing the predictions for the first fifteen items.
Deliverable:
A report summarizing the average training accuracy, average test accuracy, and insights on the F1-scores.
Task (b): Modified Hidden Layers
Objective:
Change the model to use a hidden layer configuration of 48 neurons in the first hidden layer and 24 neurons in the second hidden layer.
Adjust the output layer to 3 neurons (ensuring it uses softmax).
Run the modified program three times.
Deliverable:
A report with the average training/test accuracy and a detailed discussion on the behavior of the F1-scores, supported by predictions for the first ten items.
Task (c): Increased Epochs
Objective:
With the same settings as in Task (b), change the number of training epochs to 200.
Run the program three times.
Deliverable:
A report summarizing the average training and test accuracy along with a discussion of F1-scores, using the predictions for the first ten items as reference.
Running the Code
Dependencies:
Ensure you have the following packages installed:

numpy
pandas
scikit-learn
keras (or tensorflow.keras)
gdown
Instructions:

Comment out any cell-specific run instructions (if using a Jupyter Notebook).
Use the "Runtime" > "Run all" command (or equivalent in your environment) to execute the entire code.
Record the results from the console output for each run, especially focusing on the training/test accuracy and F1-scores.
Take screenshots of the results as evidence and include them in your final report document.
Modifications:

For Task (b), ensure that the hidden layers are set to 48 and 24 neurons respectively, and the output layer to 3 neurons.
For Task (c), update the number of epochs in the model.fit() function to 200.
Code Explanation
Data Loading and Preprocessing:
The dataset is loaded from cim-3pos.csv. Features are one-hot encoded using OneHotEncoder, while labels are encoded with LabelEncoder and converted to categorical format for multi-class classification.

Model Architecture:
The neural network uses:

An input layer corresponding to 1497 features.
Two hidden layers with ReLU activation to capture non-linear relationships.
A final softmax output layer that outputs a probability distribution over the three POS classes.
Model Evaluation:
The code calculates and prints:

Probabilities for the first few test cases.
A detailed comparison of predictions versus actual labels.
Overall performance metrics including a confusion matrix and classification report.
