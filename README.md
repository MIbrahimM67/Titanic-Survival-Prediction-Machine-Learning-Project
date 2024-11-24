# Titanic Survival Prediction with Neural Networks

This project implements a neural network model for a binary classification problem to predict survival on the Titanic. 
The model is trained, validated, and tested on a preprocessed Titanic dataset. Key features include custom gradient descent optimization, forward and backward propagation, and evaluation using metrics like accuracy.

---

## 📚 Project Overview
The goal of this project is to predict passenger survival (binary classification) using a neural network built from scratch with NumPy. The project focuses on:
- Preprocessing the Titanic dataset for missing values and feature encoding.
- Dividing the dataset into training, validation, and test sets.
- Training the model using gradient descent and monitoring loss.
- Evaluating the model on unseen test data using accuracy as the primary metric.

---

## 📊 Dataset
The dataset used is the Titanic dataset, which contains information about passengers, such as:
- **Features**: Passenger attributes (e.g., `Pclass`, `Age`, `Fare`, `Sex`).
- **Target Variable**: `Survived` (binary classification: 1 = Survived, 0 = Did not survive).

### Preprocessing Steps:
1. **Handle Missing Values**:
   - Dropped the `Cabin` column due to excessive missing data.
   - Filled missing values in the `Age` column with the mean.
   - Filled missing values in the `Embarked` column with the mode.
2. **Feature Encoding**:
   - Encoded the `Sex` column as numerical values using `LabelEncoder`.
3. **Feature Selection**:
   - Removed unnecessary columns: `Name`, `Ticket`, `PassengerId`, and `Embarked`.
4. **Data Split**:
   - Training Set: 60%
   - Validation Set: 20%
   - Test Set: 20%

---

## 🏗️ Model Architecture
The neural network has the following structure:
1. **Input Layer**: Accepts 7 features (after preprocessing).
2. **Hidden Layer**: Contains 64 neurons with the sigmoid activation function.
3. **Output Layer**: 1 neuron with the sigmoid activation function for binary classification.

---

## 🚀 Key Features
- **Gradient Descent**: Optimized weights and biases using gradient descent.
- **Forward and Backward Propagation**: Implemented manually with NumPy for a two-layer neural network.
- **Evaluation Metrics**: Accuracy calculated on the test set.

---

## 🧠 Training Details
1. **Forward Propagation**:
   - Calculated activations for the hidden and output layers using the sigmoid function.
   - Computed training and validation losses using binary cross-entropy.
2. **Backward Propagation**:
   - Calculated gradients of weights and biases for both layers.
   - Updated weights and biases using gradient descent with a learning rate of 0.21.
3. **Epochs**:
   - Trained the model for 50,000 epochs, logging losses every 1,000 epochs.

---

## 📈 Results
After training, the model was evaluated on the test set with the following result:

| Metric      | Value (%) |
|-------------|-----------|
| **Accuracy** | 81.67     |

---

## 🔍 Code Highlights
1. **Data Preprocessing**: Cleaned and prepared the Titanic dataset for training.
2. **Neural Network Training**:
   - Implemented forward propagation to compute outputs.
   - Used backward propagation to calculate gradients and optimize weights.
3. **Testing**: Evaluated the model on the test set using accuracy as the primary metric.

---

## 🛠️ Dependencies
The project uses the following libraries:
- **NumPy**: For mathematical computations.
- **Pandas**: For data handling.
- **scikit-learn**: For preprocessing and splitting datasets.
- **SciPy**: For the numerically stable sigmoid function.

Install dependencies via:
```bash
pip install numpy pandas scikit-learn scipy
