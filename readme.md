# CNN: A Neural Network for Cook Islands Māori Parts of Speech

**GooferNet** is an independent neural network project designed to predict parts of speech in Cook Islands Māori text data. It leverages a fully connected architecture (Multilayer Perceptron) along with one-hot encoding to classify tokens into three categories: noun, verb, or preposition.

## Overview

- **Objective:** Predict whether a given Cook Islands Māori token is a **noun**, **verb**, or **preposition**.
- **Methodology:** 
  - Convert tokens into high-dimensional one-hot encoded vectors.
  - Train a sequential neural network using Keras/TensorFlow.
  - Evaluate the model using accuracy, F1-scores, a classification report, and a confusion matrix.

## Project Structure
.
├── GooferNet.py            # Main script containing data loading, model creation, training, and evaluation
├── cim-3pos.csv            # Example CSV dataset with Cook Islands Māori tokens and their POS tags
├── readme.md               # You're reading it now!



> **Note:** The project uses the `cim-3pos.csv` dataset. If you are using a different dataset, adjust the filename and column names accordingly.

## Data Preparation

1. **Dataset**  
   - The CSV file (`cim-3pos.csv`) contains tokens and their corresponding parts of speech.
   - Modify the file or code if your dataset uses different column names.

2. **One-Hot Encoding**  
   - Uses `OneHotEncoder` from `sklearn` to transform tokens into binary feature vectors.
   - Ensures each unique token is represented by a unique binary vector.

3. **Train/Test Split**  
   - The dataset is divided into 90% training data and 10% test data using `train_test_split` from `sklearn`.

## Model Architecture

**GooferNet** employs a simple feedforward neural network with:

- **Input Layer:** Dimension equals the number of unique tokens after one-hot encoding.
- **Hidden Layer 1:** 48 neurons with ReLU activation.
- **Hidden Layer 2:** 24 neurons with ReLU activation.
- **Output Layer:** 3 neurons (for the three POS classes) with softmax activation.

The network is compiled with the `adam` optimizer and uses `categorical_crossentropy` as the loss function.

### Model Summary

Layer (type)               Output Shape              Param #  
=================================================================
dense (Dense)              (None, 48)                <calculated automatically>
dense_1 (Dense)            (None, 24)                <calculated automatically>
dense_2 (Dense)            (None, 3)                 <calculated automatically>
=================================================================

## How to Use

### 1. Install Dependencies

Ensure you have the necessary libraries installed. You can install them using:


python CNN.py



## Sample Output:
===== Accuracy of test set =====
87.0%

===== Classification Report =====
              precision    recall  f1-score   support
0 (noun)         ...
1 (verb)         ...
2 (prep)         ...

===== Confusion Matrix =====
[[TP  FP ...]
 [FN  TP ...]
 [...       ]]



