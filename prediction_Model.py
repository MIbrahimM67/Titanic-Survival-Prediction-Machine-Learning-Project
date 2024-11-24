import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.special import expit  # Numerically stable sigmoid function
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Corrected file path
titanic_data = pd.read_csv(r"D:/programming 2023/AI/Titanic_Prediction/train.csv")

# Data Preprocessing
print(titanic_data.isnull().sum())

# Drop 'Cabin' column since it has the most null values
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# Fill missing values in 'Age' with the mean
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# Fill missing values in 'Embarked' with the mode
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

print(titanic_data.isnull().sum())

# Encode categorical column 'Sex'
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])

# Drop unnecessary columns
titanic_data.drop(['Name', 'Ticket', 'PassengerId', 'Embarked'], axis=1, inplace=True)

# Split data into features (X) and target (Y)
X = titanic_data.drop('Survived', axis=1)
Y = titanic_data['Survived']

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Neural network parameters
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 1
learning_rate = 0.21
epochs = 50000
m = X_train.shape[0]  # Number of training samples

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * 0.01  # Input to hidden weights
b1 = np.zeros((1, hidden_dim))  # Hidden layer bias
W2 = np.random.randn(hidden_dim, output_dim) * 0.01  # Hidden to output weights
b2 = np.zeros((1, output_dim))  # Output layer bias

# Training loop
for epoch in range(epochs):
    # Forward pass (use X_train for training)
    Z1 = np.dot(X_train, W1) + b1
    h1 = expit(Z1)
    Z2 = np.dot(h1, W2) + b2
    A2 = expit(Z2)

    # Compute training loss
    train_loss = log_loss(y_train.values.reshape(-1, 1), A2)

    # Compute validation loss
    Z1_val = np.dot(X_val, W1) + b1
    h1_val = expit(Z1_val)
    Z2_val = np.dot(h1_val, W2) + b2
    A2_val = expit(Z2_val)
    val_loss = log_loss(y_val.values.reshape(-1, 1), A2_val)

    # Gradients for output layer
    dA2 = A2 - y_train.values.reshape(-1, 1)
    dZ2 = dA2 * (A2 * (1 - A2))  # Derivative of sigmoid
    dW2 = np.dot(h1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Gradients for hidden layer
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (h1 * (1 - h1))  # Derivative of sigmoid
    dW1 = np.dot(X_train.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Update weights and biases
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Print loss every 100 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


#lets do the test part now
# Forward pass on test data
Z1_test = np.dot(X_test, W1) + b1 #using X-test now 
h1_test = expit(Z1_test)  # Activation for hidden layer
Z2_test = np.dot(h1_test, W2) + b2
A2_test = expit(Z2_test)  # Predicted probabilities

# Convert probabilities to binary class predictions (threshold = 0.5)
y_pred = (A2_test >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(y_pred.flatten() == y_test.values)
print(f"Test Accuracy: {accuracy * 100:.2f}%")