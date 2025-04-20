import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


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

# performs multiple forward passes through the model with dropout enabled generating a distribution of predictions for the data in the loader
def predictions_with_uncertainties(model, data_loader, n_samples=100):
    model.eval()
    model.enable_mc_dropout()
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            predicts = []
            for X_batch, _ in data_loader:
                y_preds = model(X_batch)
                predicts.append(y_preds.numpy())
            predictions.append(np.vstack(predicts))
    predictions = np.stack(predictions)
    mean_predictions = np.mean(predictions, axis = 0) #point predictions
    std_predictions = np.std(predictions, axis = 0) # epistemic uncertainty
    return mean_predictions, std_predictions


#parameters found through bayesian optimisation using optuna
best_params = {'hidden_dim': 57,'lr': 0.00026871333611041425, 'dropout_rate': 0.12757236756343715,'batch_size': 16}


r2_scores = []
mean_squared_error_scores = []
k_fold = KFold(n_splits=5, shuffle=True,random_state=1)


#outer loop is for cross validation, to validate model performance more robustly
for fold, (train_index, val_index) in enumerate(k_fold.split(X_scaled)):
    X_train_cv, X_val_cv = X_scaled.iloc[train_index], X_scaled.iloc[val_index]
    y_train_cv, y_val_cv = y_scaled.iloc[train_index], y_scaled.iloc[val_index]
    train_dataset = NNDataset(X_train_cv, y_train_cv)
    validation_dataset = NNDataset(X_val_cv,y_val_cv)
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)#shuffling for training
    validation_loader = DataLoader(validation_dataset,batch_size=best_params['batch_size'], shuffle=False) #no shuffle for testing

    model = MCPytorchNN(input_dim=X_train_cv.shape[1], output_dim=y_train_cv.shape[1],hidden_dim=best_params['hidden_dim'],dropout_rate=best_params['dropout_rate'])
    optimizer = optim.Adam(model.parameters(), lr = best_params['lr'])
    best_validation_loss = float('inf') #initiates the validation loss at infinity
    patience = 20
    patience_tracker = 0 # keepsing track of how many epochs have passed without improvement in the validation loss
    num_epochs = 300

    #training loop
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            y_predict = model(X_batch)
            loss = model.loss_function(y_predict, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #early stopping validation
        model.eval()
        validation_predictions = []
        validation_true = []
        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                y_predict = model(X_batch)
                validation_predictions.append(y_predict.numpy())
                validation_true.append(y_batch.numpy())

        y_pred_val = np.vstack(validation_predictions)
        y_true_val = np.vstack(validation_true)

        y_pred_val_orig = y_scaler.inverse_transform(y_pred_val)
        y_true_val_orig = y_scaler.inverse_transform(y_true_val)

        val_loss = mean_squared_error(y_true_val_orig, y_pred_val_orig)

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            patience_counter = 0

            #save model checkpoint for this fold
            model_path = f"model_fold_{fold + 1}.pt"
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    #final model eval using MC dropout
    mc_preds, _ = predictions_with_uncertainties(model, validation_loader, n_samples=20)
    y_val_true = np.vstack([y.numpy() for _, y in validation_loader])

    y_val_pred_orig = y_scaler.inverse_transform(mc_preds)
    y_val_true_orig = y_scaler.inverse_transform(y_val_true)

    fold_mse = mean_squared_error(y_val_true_orig, y_val_pred_orig)
    fold_r2 = r2_score(y_val_true_orig, y_val_pred_orig)

    mean_squared_error_scores.append(fold_mse)
    r2_scores.append(fold_r2)

    print(f"  Fold MSE: {fold_mse:.4f}, Fold R²: {fold_r2:.4f}")

#-------------------------------------------------------------------------------------------#

#final evaluation
mean_mse = np.mean(mean_squared_error_scores)
mean_r2 = np.mean(r2_scores)

print("\nFinal Cross-Validated Performance:")
print(f"  Mean MSE: {mean_mse:.4f}")
print(f"  Mean R²: {mean_r2:.4f}")





