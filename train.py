import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import SWEDataLoader
from modelzoo.NeuralNet import SWE_Net
from sklearn.metrics import mean_squared_error
import numpy as np
from utils.SpatialTransform import SpatialTransformer
from utils.backtransform import back_transform_predictions
import yaml
from types import SimpleNamespace

def train_model(cfg: SimpleNamespace) -> SWE_Net:
    loader = SWEDataLoader(cfg)
    train_loader, val_loader, transformer, spatial_objs = loader.prepare()

    device = torch.device(cfg.device)
    cfg.device = device
    
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["dynamic forcing"].shape[-1]
    
    model = SWE_Net(input_dim=input_dim, hidden_dims=cfg.hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    best_rmse = float("inf")

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss = 0
        n_train_samples = 0
        
        for batch in train_loader:
            X = batch["dynamic forcing"].to(device)  
            y = batch["swe"].to(device)        
            
            preds = model(X)
            loss = criterion(preds, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X.size(0)
            n_train_samples += X.size(0)

        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                X = batch["dynamic forcing"].to(device)   
                y = batch["swe"]                           
                dates_batch = batch["dates"]               
                preds = model(X).cpu().numpy()            
                B, T = preds.shape

                if hasattr(dates_batch, "tolist"):
                    dates_batch = dates_batch.tolist()

                for i in range(B):
                    dates_i = dates_batch[i]
                    try:
                        dates_i = [str(d) for d in dates_i]
                    except Exception:
                        pass
                    for t in range(T):
                        dt = dates_i[t]  # e.g., 'YYYY-MM-DD'
                        preds[i, t] = back_transform_predictions(
                            np.array([preds[i, t]]), transformer, spatial_objs[dt]
                        )[0]

                val_preds.append(preds)
                val_targets.append(y.numpy())
        y_pred = np.concatenate(val_preds)
        y_true = np.concatenate(val_targets)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"Epoch {epoch+1}/{cfg.n_epochs} | Train Loss: {epoch_loss/n_train_samples:.4f} | Val RMSE: {rmse:.4f}")

        if rmse < best_rmse and cfg.save:
            best_rmse = rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': rmse,
            }, cfg.save_path)

    return model

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    model = train_model(cfg)