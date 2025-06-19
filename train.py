import torch
import torch.nn as nn
import torch.optim as optim
from config import SWEConfig
from dataloader import SWEDataLoader
from modelzoo.NeuralNet import SWE_Net
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(cfg: SWEConfig): #Matches practice of passing train_loader into SWEConfig from NeuralHydrology
    loader = SWEDataLoader(cfg)
    train_loader, val_loader, transformer, val_df = loader.prepare() #references dataloader to apply SpatialTransformer

    #--Initialization--
    device = torch.device(cfg.device)
    input_dim = next(iter(train_loader))[0].shape[1]
    model = SWE_Net(input_dim=input_dim, hidden_dims=cfg.hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    best_rmse = float("inf") 

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss = 0
        
        #training loop - below is adjusted to make device compatible 
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)

        #validation loop
        model.eval()
        preds_all, y_all = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                preds_all.append(model(X).cpu().numpy())
                y_all.append(y.numpy())

        #metrics (printing training loss to make sure model is at least running correctly)       
        y_pred = np.concatenate(preds_all).flatten()
        y_true = np.concatenate(y_all)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"Epoch {epoch+1}/{cfg.n_epochs} | Train Loss: {epoch_loss/len(train_loader.dataset):.4f} | Val RMSE: {rmse:.4f}")

        #save best model weights
        if rmse < best_rmse and cfg.save:
            best_rmse = rmse
            torch.save(model.state_dict(), cfg.save_path)

    print("Training complete.") #checkpoint
    return model

#direct run entry point w/ cfg setup
if __name__ == "__main__":
    cfg = SWEConfig(
        swe_path="data/SWE_data_CONUS.csv",
        meta_path="data/Snotel_Locations_Filtered_v3.csv",
        sample_date="1/5/2000"
    )
    train_model(cfg)