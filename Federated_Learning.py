import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('data/preprocessed_training_UNSW_NB15.csv')

# Separate features ,labels and normalize the data


def preprocess_data(filepath):

    # Separate features and labels
    data = pd.read_csv(filepath)
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # Handle class imbalance (if needed)
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    X_minority = X[y == 1]
    y_minority = y[y == 1]

    X_minority_upsampled, y_minority_upsampled = resample(
        X_minority, y_minority,
        replace=True,
        n_samples=len(y_majority),
        random_state=42
    )

    X_balanced = np.vstack((X_majority, X_minority_upsampled))
    y_balanced = np.hstack((y_majority, y_minority_upsampled))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


filepath = 'data/preprocessed_training_UNSW_NB15.csv'
X_train, X_test, y_train, y_test = preprocess_data(filepath)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training label shape: {y_train.shape}")
print(f"Testing label shape: {y_test.shape}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Convert all tensors to float
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()


def create_client_data(num_clients=3):
    """Simulates datasets for clients."""
    client_data = []
    for _ in range(num_clients):
        client_data.append((X_train, y_train))
    return client_data

# X_train[1].shape[0] = 36 # Number of features

# Define a simple ANN  model for clients but we can make it more complex


class clientsMod(nn.Module):
    def __init__(self, input_dim=36):
        super(clientsMod, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Define a simple ANN model for the server but we can make it more complex


class MainModel(nn.Module):
    def __init__(self, input_dim=36):
        super(MainModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def federated_avg(models):
    """Aggregate model weights by averaging."""
    avg_model = models[0]
    with torch.no_grad():
        for key in avg_model.state_dict():
            avg_model.state_dict()[key].copy_(
                sum(model.state_dict()[key] for model in models) / len(models)
            )
    return avg_model


# Training function for clients

def train_client(model, data, epochs=5, lr=0.01):
    """Train a model on a single client's data."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    x, y = data
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = model(batch_x).squeeze()
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()

    return model

# Evaluation function


def evaluate_model(model, data):
    """Evaluate model accuracy on a dataset."""
    model.eval()
    x, y = data
    with torch.no_grad():
        y_pred = model(x).squeeze()
        # y_pred = (y_pred > 0.5).float()  # Threshold to get binary predictions
        accuracy = (y_pred == y).float().mean().item()
    return accuracy


def plot_accuracy_graph(global_epochs, global_accuracies, local_accuracies):
    """
    Plots a graph of epochs vs accuracy for global and local models.

    Args:
        global_epochs (int): Total number of global epochs.
        global_accuracies (list): List of global accuracies over the epochs.
        local_accuracies (list of lists): List of local accuracies for each client.
    """
    epochs = np.arange(1, global_epochs + 1)

    # Plot global accuracy
    plt.plot(epochs, global_accuracies,
             label='Global Accuracy', marker='o', color='b')

    # Plot local accuracies for each client
    for i, local_acc in enumerate(local_accuracies):
        plt.plot(epochs, local_acc,
                 label=f'Client {i + 1} Accuracy', linestyle='--')

    # Labels and title
    plt.xlabel('Global Epochs')
    plt.ylabel('Accuracy')
    plt.title('Federated Learning: Epochs vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def federated_learning_simulation(num_clients=3, global_epochs=10, local_epochs=50):
    input_dim = 36  # Features
    client_data = create_client_data(
        num_clients=num_clients)

    # Initialize a model for each client
    client_models = [clientsMod(input_dim) for _ in range(num_clients)]

    # Global model
    global_model = MainModel(input_dim)

    global_accuracies = []
    # To store local accuracies per client
    local_accuracies = [[] for _ in range(num_clients)]

    for global_epoch in range(global_epochs):
        print(f"Global Epoch {global_epoch + 1}/{global_epochs}")

        # Step 1: Send global model to clients
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        # Step 2: Train local models and collect accuracies
        for i, model in enumerate(client_models):
            train_client(model, client_data[i], epochs=local_epochs)
            local_acc = evaluate_model(model, client_data[i])
            # Store local accuracy for each client
            local_accuracies[i].append(local_acc)
            print(f"  Client {i + 1} Local Accuracy: {local_acc:.2f}")

        # Step 3: Aggregate models
        global_model = federated_avg(client_models)

        # Step 4: Evaluate global model on all clients
        global_acc = np.mean(
            [evaluate_model(global_model, client_data[i])
             for i in range(num_clients)]
        )
        print(f"  Global Model Accuracy: {global_acc:.2f}\n")

        # Store global accuracy
        global_accuracies.append(global_acc)

    # Plot the accuracy graph after the training loop
    plot_accuracy_graph(global_epochs, global_accuracies, local_accuracies)

    return global_model


federated_learning_simulation()
