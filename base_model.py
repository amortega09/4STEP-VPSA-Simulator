import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd 
from sklearn.preprocessing import StandardScaler


inputs = pd.read_excel('inputs_all.xlsx')
outputs = pd.read_excel('outputs_all.xlsx')

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = pd.DataFrame(X_scaler.fit_transform(inputs), columns=inputs.columns)
y_scaled = pd.DataFrame(y_scaler.fit_transform(outputs), columns=outputs.columns)


joblib.dump(X_scaler, "1X_scaler.pkl")
joblib.dump(y_scaler, "2y_scaler.pkl")


#setting random seed to ensure consistency

torch.manual_seed(1)
np.random.seed(1)

#create dataset and convert to torch tensors
class NNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype = torch.float32)
        self.y = torch.tensor(y.values, dtype = torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
    

# neural network with MC dropout wrapped in feature extractor for smoother code
    
class MCPytorchNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_rate):
        super(MCPytorchNN, self).__init__()
        self.mc_dropout = False 

        self.feature_layers = nn.Sequential(nn.Linear(input_dim,hidden_dim),
        nn.ReLU(), nn.Dropout(dropout_rate),nn.Linear(hidden_dim,hidden_dim), nn.ReLU(),
        nn.Dropout(dropout_rate))

        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.log_variance = nn.Parameter(torch.zeros(output_dim))


    def forward(self, x):
        def apply_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        if self.mc_dropout:
            self.feature_layers.apply(apply_dropout)

        features = self.feature_layers(x)
        return self.final_layer(features)
    
    #the function below implements heteroscedastic loss, accounting for both prediction error and model uncertainty
    def loss_function(self, y_pred, y_true):
        var = torch.exp(self.log_variance)
        loss = 0.5 * torch.sum((y_pred - y_true)**2 / var.unsqueeze(0) + torch.log(var)) / y_pred.size(0)
        return loss
    
    def enable_mc_dropout(self):
        self.mc_dropout = True


#----------------------------------------#
#Training the final model on full dataset to improve accuracy #
#----------------------------------------#

#parameters found through bayesian optimisation using optuna
best_params = {'hidden_dim': 57,'lr': 0.00026871333611041425, 'dropout_rate': 0.12757236756343715,'batch_size': 16}

# create dataset and dataloader from full data
full_dataset = NNDataset(X_scaled, y_scaled)
full_loader = DataLoader(full_dataset, batch_size=best_params['batch_size'], shuffle=True)

model = MCPytorchNN(
    input_dim=X_scaled.shape[1],
    output_dim=y_scaled.shape[1],
    hidden_dim=best_params['hidden_dim'],
    dropout_rate=best_params['dropout_rate']
)

optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

#train on full data
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in full_loader:
        y_predict = model(X_batch)
        loss = model.loss_function(y_predict, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# save final trained model
torch.save(model.state_dict(), "final_model.pt")

print("Final model trained on full dataset and saved as 'final_model.pt'")
