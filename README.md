## Privacy - Preserving Threat Detection using Federated Learning

This project focuses on building a privacy-preserving threat detection system using Federated Learning (FL). The goal is to develop a machine-learning model capable of identifying security threats without the need to share sensitive data between devices or organizations. Federated Learning allows model training on decentralized data, ensuring that private information remains local while still enabling the collaborative training of a global model.

### Dataset Explanation:

We used **UNSW_NB15** dataset

**Dataset Size:** 70k 

The raw network packets of the UNSW-NB 15 dataset was created by the IXIA PerfectStorm tool in the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS) for generating a hybrid of real modern normal activities and synthetic contemporary attack behaviours. This dataset has nine types of attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms. The Argus, and Bro-IDS tools are used and twelve algorithms are developed to generate totally 49 features with the class label.

[![Dataset Link]()](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15?select=UNSW_NB15_training-set.csv)

These are the columns 

['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
       'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
       'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
       'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
       'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
       'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label']

### Tools Used

`torch`
`torchvision` 
`matplotlib` 
`pandas` 
`scikit-learn`


**Data Preprocessing:**

`data_preprocessing.py` - The data is pre-processed for training the model

`main code`

## This is for Nvidia GPU
```bash
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)
```

## This is for Mac OS
```bash
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)
```

The pre-processed, normalized data is split into train and test data. These data will be in array form so we need to convert it into tensor to train the model. 

```bash
import torch

# Convert all tensors to float
X_train = torch.tensor(X_train).float().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
y_test = torch.tensor(y_test).float().to(device)
```

We have taken 3 virtual clients.

```bash
def create_client_data(num_clients=3):
    """Simulates datasets for clients."""
    client_data = []
    for _ in range(num_clients):
        client_data.append((X_train, y_train))
    return client_data
```

Here in this example, we have taken the same dataset for all three clients but it can be different in real-world usage.

## Neural Network Model

Here all three clients have different neural network models. In our case, we took a simple neural network for all three but we can use RNN also.

There will be one neural network model for the server but In our case, we took a simple neural network for all three but we can use RNN also.

```bash
import torch.nn as nn

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
```

![image](https://github.com/user-attachments/assets/77a7b0ac-2907-4102-8aac-928112386f4f)

**Federated Learning**

The client model will be trained individually on their own dataset. 

The server model is trained using the average of all client's model parameters, not with the actual data of each individual client. 

Therefore the data will be private to the actual client which increases the security.

```bash
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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
```

Combining everything together

```bash

def federated_learning_simulation(num_clients=5, global_epochs=10, local_epochs=10):
    input_dim = 36  # Features
    client_data = create_client_data(
        num_clients=num_clients)

    # Initialize a model for each client
    client_models = [clientsMod(input_dim).to(device) for _ in range(num_clients)]

    # Global model
    global_model = MainModel(input_dim).to(device)

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
            # print(f"  Client {i + 1} Local Accuracy: {local_acc:.2f}")

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
```
