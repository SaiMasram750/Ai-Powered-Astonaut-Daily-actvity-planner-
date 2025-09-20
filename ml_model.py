"""
This module provides functions to simulate astronaut task data, train a machine learning
model to predict task priority based on the simulated data, and a function to predict
the priority of a new task using the trained model.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def simulate_data(num_rows=100):
    """
    Simulates a dataset of astronaut tasks with various attributes and a calculated priority.

    Args:
        num_rows (int): The number of rows (tasks) to simulate. Defaults to 100.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated task data with columns
                      'task_description', 'urgency', 'complexity', 'time_sensitive',
                      'assigned_to', and 'priority'.
    """
    np.random.seed(42) # for reproducibility
    task_descriptions = [
        "Repair solar panel", "Analyze rock sample", "Conduct EVA",
        "Monitor life support", "Prepare meal", "Exercise",
        "Communicate with Earth", "Perform experiment", "Maintain equipment",
        "Update logs"
    ]
    urgencies = ["High", "Medium", "Low"]
    complexities = ["High", "Medium", "Low"]
    time_sensitivities = [True, False]
    assigned_tos = ["Astronaut A", "Astronaut B", "Astronaut C"]

    data = {
        'task_description': np.random.choice(task_descriptions, num_rows),
        'urgency': np.random.choice(urgencies, num_rows, p=[0.4, 0.4, 0.2]),
        'complexity': np.random.choice(complexities, num_rows, p=[0.3, 0.4, 0.3]),
        'time_sensitive': np.random.choice(time_sensitivities, num_rows, p=[0.3, 0.7]),
        'assigned_to': np.random.choice(assigned_tos, num_rows, p=[0.4, 0.3, 0.3])
    }

    df_tasks = pd.DataFrame(data)

    # Determine priority based on a simple rule
    def determine_priority(row):
        if row['urgency'] == 'High' or (row['urgency'] == 'Medium' and row['time_sensitive']):
            return 'High'
        elif row['complexity'] == 'High' and row['urgency'] == 'Medium':
            return 'High'
        elif row['urgency'] == 'Medium' or row['complexity'] == 'Medium':
            return 'Medium'
        else:
            return 'Low'

    df_tasks['priority'] = df_tasks.apply(determine_priority, axis=1)

    return df_tasks

# Simulate the data for training
df_tasks = simulate_data(200)

# --- Data Preparation ---
# Separate features (X) and target (y)
X = df_tasks.drop('priority', axis=1)
y = df_tasks['priority']

# Define categorical features for one-hot encoding
categorical_features = ['task_description', 'urgency', 'complexity', 'assigned_to']
# Convert categorical features using one-hot encoding
X_processed = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Define mapping for target variable 'priority' to numerical labels
priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
# Convert target variable 'priority' to numerical labels
y_processed = y.map(priority_mapping)

# --- Model Training ---
# Initialize the Decision Tree Classifier model
model = DecisionTreeClassifier(random_state=42)

# Train the model using the processed features and target
trained_model = model.fit(X_processed, y_processed)

def predict_priority(task_description, urgency, complexity, time_sensitive, assigned_to):
    """
    Predicts the priority of a task based on its details using the trained model.

    Args:
        task_description (str): Description of the task.
        urgency (str): Urgency level of the task ("Low", "Medium", "High").
        complexity (str): Complexity level of the task ("Low", "Medium", "High").
        time_sensitive (bool): Whether the task is time-sensitive.
        assigned_to (str): The astronaut assigned to the task.

    Returns:
        str: The predicted priority ("Low", "Medium", or "High").
    """
    # Create a DataFrame from the input task details
    input_data = pd.DataFrame([{
        'task_description': task_description,
        'urgency': urgency,
        'complexity': complexity,
        'time_sensitive': time_sensitive,
        'assigned_to': assigned_to
    }])

    # Apply the same preprocessing steps as the training data
    # Use pd.get_dummies to one-hot encode categorical features
    input_processed = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)

    # Ensure the input DataFrame has the same columns as the training data after one-hot encoding
    # This handles cases where a specific category from the training data is not in the input
    missing_cols = set(X_processed.columns) - set(input_processed.columns)
    for c in missing_cols:
        input_processed[c] = 0
    # Ensure the order of columns in the input matches the training data for correct prediction
    input_processed = input_processed[X_processed.columns]

    # Predict the numerical priority label using the trained model
    predicted_label = trained_model.predict(input_processed)[0]

    # Define the reverse mapping from numerical labels back to categorical priority strings
    reverse_priority_mapping = {v: k for k, v in priority_mapping.items()}
    # Map the numerical prediction back to the categorical priority string
    predicted_priority = reverse_priority_mapping[predicted_label]

    return predicted_priority

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Retrain the model on the training data
model = DecisionTreeClassifier(random_state=42)
trained_model = model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = trained_model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Retrain the model on the training data
model = DecisionTreeClassifier(random_state=42)
trained_model = model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = trained_model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
