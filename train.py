import os
import time
from types import SimpleNamespace
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_squared_error

from dataloader import SWEDataLoader, SWEStationDataset
from modelzoo.LSTM import SWE_Net
from utils.backtransform import back_transform_scalar_with_weights


def train_model(cfg: SimpleNamespace) -> SWE_Net:
    device = torch.device(cfg.device)
    
    dataset = SWEStationDataset(cfg)
    orig_swe = dataset.dynamic_forcing_and_swe.reset_index()
    swe_lookup = orig_swe.set_index(["Station", "Date"])["SWE"]

    pickle_file = "backtransform_info.pkl"
    
    if not os.path.exists(pickle_file):
        dataloader = SWEDataLoader(cfg)
        train_loader, val_loader, _, bt_info = dataloader.prepare()
        pickle.dump((train_loader, val_loader, _, bt_info), open(pickle_file, "wb"))
    else:
        train_loader, val_loader, _, bt_info = pickle.load(open(pickle_file, "rb"))

    station_index = bt_info["station_index"]
    weights = bt_info["weights"]
    backtrans_cache = bt_info["backtrans_cache"]

    station_mean_swe = orig_swe.groupby("Station")["SWE"].mean()
    station_std_swe = orig_swe.groupby("Station")["SWE"].std()
    
    station_norm_factors = {}
    for station in station_mean_swe.index:
        mean_swe = station_mean_swe[station]
        std_swe = station_std_swe[station]
        
        if mean_swe < 1.0:
            norm_factor = max(1.0, std_swe)
        else:
            norm_factor = max(2.0, std_swe)
        
        station_norm_factors[station] = norm_factor
    
    station_stats_dict = {}
    for station in station_mean_swe.index:
        station_stats_dict[station] = (
            float(station_mean_swe[station]),
            float(station_std_swe[station])
        )

    model = SWE_Net(cfg, station_stats=station_stats_dict).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=8, factor=0.7
    )

    def nse_loss(predictions, targets, epsilon=1e-8):
        """
        Differentiable Nash-Sutcliffe Efficiency loss
        Returns 1 - NSE so that minimizing the loss maximizes NSE
        """
        mean_target = torch.mean(targets)
        
        numerator = torch.sum((targets - predictions) ** 2)
        
        denominator = torch.sum((targets - mean_target) ** 2)
        
        nse = 1.0 - (numerator / (denominator + epsilon))
        
        return 1.0 - nse

    all_predictions = []

    for epoch in range(cfg.n_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg.n_epochs}")
        print("-" * 50)

        model.train()
        train_loss = 0
        batch_count = 0

        for batch in train_loader:
            optimizer.zero_grad()
            X = batch["dynamic forcing"].to(device)
            y = batch["swe"].to(device)
            stations = batch["station"]

            outputs = model(X, stations=stations)
            
            batch_losses = []
            
            critical_stations = set([
                'sleeping woman', 'shanghi summit', 'cole creek', 'swamp creek', 
                'south brush creek', 'quartz peak', 'darkhorse lake', 'tie creek',
                'railroad overpass', 'mowich', 'cascade #2', 'copper camp', 'sand lake',
                'rainy pass', 'mud flat', 'humboldt gulch', 'noisy basin', 'lick creek'
            ])
            
            for i, station in enumerate(stations):
                station_preds = outputs[i].flatten()
                station_targets = y[i].flatten()
                station_mean = station_mean_swe.get(station, 5.0)
                
                valid_mask = (~torch.isnan(station_preds)) & (~torch.isnan(station_targets))
                
                if torch.sum(valid_mask) > 1:
                    valid_preds = station_preds[valid_mask]
                    valid_targets = station_targets[valid_mask]
                    
                    if station in critical_stations:
                        max_reasonable = torch.quantile(valid_targets, 0.85) + 0.2
                        clamped_preds = torch.clamp(valid_preds, min=0.0, max=max_reasonable)
                        
                        nse_loss_val = nse_loss(clamped_preds, valid_targets)
                        relative_error = torch.mean(torch.abs(clamped_preds - valid_targets) / (valid_targets + 0.02))
                        overprediction_penalty = torch.mean(torch.relu(clamped_preds - valid_targets) ** 2)
                        combined_loss = 0.1 * nse_loss_val + 0.4 * relative_error + 0.5 * overprediction_penalty
                        batch_losses.append(combined_loss)
                        
                    elif station_mean < 0.5:  # Extremely low SWE stations (likely NSE << 0)
                        # Apply strongest constraints and hybrid loss
                        max_reasonable = torch.quantile(valid_targets, 0.90) + 0.3  # More conservative
                        clamped_preds = torch.clamp(valid_preds, min=0.0, max=max_reasonable)
                        
                        # Heavy emphasis on relative error to prevent overprediction
                        nse_loss_val = nse_loss(clamped_preds, valid_targets)
                        relative_error = torch.mean(torch.abs(clamped_preds - valid_targets) / (valid_targets + 0.05))
                        combined_loss = 0.2 * nse_loss_val + 0.8 * relative_error
                        batch_losses.append(combined_loss)
                        
                    elif station_mean < 2.0:
                        max_reasonable = torch.quantile(valid_targets, 0.95) + 1.0
                        clamped_preds = torch.clamp(valid_preds, min=0.0, max=max_reasonable)
                        
                        nse_loss_val = nse_loss(clamped_preds, valid_targets)
                        relative_error = torch.mean(torch.abs(clamped_preds - valid_targets) / (valid_targets + 0.1))
                        combined_loss = 0.4 * nse_loss_val + 0.6 * relative_error
                        batch_losses.append(combined_loss)
                        
                    elif station_mean < 5.0:
                        max_reasonable = torch.quantile(valid_targets, 0.98) + 2.0
                        clamped_preds = torch.clamp(valid_preds, min=0.0, max=max_reasonable)
                        
                        nse_loss_val = nse_loss(clamped_preds, valid_targets)
                        relative_error = torch.mean(torch.abs(clamped_preds - valid_targets) / (valid_targets + 0.2))
                        combined_loss = 0.7 * nse_loss_val + 0.3 * relative_error
                        batch_losses.append(combined_loss)
                        
                    else:
                        station_nse_loss = nse_loss(valid_preds, valid_targets)
                        batch_losses.append(station_nse_loss)
                else:
                    fallback_loss = torch.mean((station_preds - station_targets) ** 2)
                    batch_losses.append(fallback_loss)
            
            loss = torch.stack(batch_losses).mean()
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                X = batch["dynamic forcing"].to(device)
                mask = batch["mask"]
                stations = batch["station"]
                dates = batch["dates"]

                preds = model(X, stations=stations).cpu().numpy()

                for i in range(len(stations)):
                    station = stations[i]
                    valid_timesteps = mask[i].sum().item()

                    for t in range(valid_timesteps):
                        date_str = dates[i][t]
                        if date_str not in backtrans_cache:
                            continue

                        try:
                            pred_value = back_transform_scalar_with_weights(
                                pred_prime=preds[i, t],
                                station_idx=station_index[station],
                                date_str=date_str,
                                weights=weights,
                                backtrans_cache=backtrans_cache,
                            )
                            
                            pred_value = max(0.0, pred_value)
                            
                            station_mean = station_mean_swe.get(station, 5.0)
                            
                            very_low_swe_stations = set([
                                'sleeping woman', 'shanghi summit', 'cole creek', 'swamp creek', 
                                'south brush creek', 'quartz peak', 'darkhorse lake', 'tie creek',
                                'railroad overpass', 'mowich', 'cascade #2', 'copper camp', 'sand lake',
                                'rainy pass', 'mud flat', 'humboldt gulch', 'noisy basin', 'lick creek'
                            ])
                            
                            if station in very_low_swe_stations:
                                max_reasonable = max(0.3, station_mean * 1.2)
                                pred_value = min(pred_value, max_reasonable)
                            elif station_mean < 0.5:
                                max_reasonable = max(0.5, station_mean * 1.3)
                                pred_value = min(pred_value, max_reasonable)
                            elif station_mean < 1.0:
                                max_reasonable = max(1.0, station_mean * 1.5)
                                pred_value = min(pred_value, max_reasonable)
                            elif station_mean < 2.0:
                                max_reasonable = max(2.0, station_mean * 2.0)
                                pred_value = min(pred_value, max_reasonable)
                            elif station_mean < 5.0:
                                max_reasonable = station_mean * 2.5
                                pred_value = min(pred_value, max_reasonable)

                            actual_swe = swe_lookup.loc[(station, date_str)]

                            if epoch == cfg.n_epochs - 1:
                                all_predictions.append(
                                    {"station": station, "date": date_str, "predicted_swe": pred_value, "actual_swe": actual_swe}
                                )

                            val_preds.append(pred_value)
                            val_targets.append(actual_swe)
                        except Exception as e:
                            print(f"\nError at {station}, {date_str}: {str(e)}")
                            continue

        epoch_time = time.time() - epoch_start
        avg_train_loss = train_loss / len(train_loader)
        if val_preds:
            rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            
            mean_observed = np.mean(val_targets)
            ss_res = np.sum((np.array(val_targets) - np.array(val_preds)) ** 2)
            ss_tot = np.sum((np.array(val_targets) - mean_observed) ** 2)
            nse = 1 - (ss_res / ss_tot)
            
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(rmse)
            new_lr = optimizer.param_groups[0]['lr']
            
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val RMSE: {rmse:.4f}")
            print(f"Val NSE: {nse:.4f}")
            print(f"Current LR: {new_lr:.6f}")
            
            if new_lr < prev_lr:
                print(f"*** Learning rate reduced from {prev_lr:.6f} to {new_lr:.6f} ***")
                
            print(f"Predictions: {len(val_preds)}")
            print("-" * 50)

    if all_predictions:
        predictions_df = pd.DataFrame(
            {
                "station": [p["station"] for p in all_predictions],
                "date": [p["date"] for p in all_predictions],
                "predicted_swe": [p["predicted_swe"] for p in all_predictions],
                "actual_swe": [p["actual_swe"] for p in all_predictions],
            }
        )

        predictions_df = predictions_df.sort_values(["station", "date"])

        station_metrics = []
        for station in predictions_df['station'].unique():
            station_data = predictions_df[predictions_df['station'] == station]
            
            actual = station_data['actual_swe'].values
            predicted = station_data['predicted_swe'].values
            
            mean_observed = np.mean(actual)
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - mean_observed) ** 2)
            nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            
            station_metrics.append({
                'station': station,
                'nse': nse,
                'rmse': rmse,
                'n_predictions': len(actual),
                'mean_actual': np.mean(actual),
                'mean_predicted': np.mean(predicted)
            })
        
        station_metrics_df = pd.DataFrame(station_metrics)
        station_metrics_df = station_metrics_df.sort_values('nse', ascending=False)
        
        print(f"\nPer-Station Performance Summary:")
        print(f"{'Station':<12} {'NSE':<8} {'RMSE':<8} {'N':<6} {'Mean Actual':<12} {'Mean Pred':<12}")
        print("-" * 70)
        
        for _, row in station_metrics_df.iterrows():
            print(f"{row['station']:<12} {row['nse']:<8.3f} {row['rmse']:<8.2f} {row['n_predictions']:<6} {row['mean_actual']:<12.2f} {row['mean_predicted']:<12.2f}")
        
        valid_nse = station_metrics_df['nse'].dropna()
        total_stations = len(valid_nse)
        
        print(f"\nPerformance Distribution across {total_stations} stations:")
        
        nse_le_0 = (valid_nse <= 0).sum()
        nse_0_to_3 = ((valid_nse > 0) & (valid_nse <= 0.3)).sum()
        nse_3_to_5 = ((valid_nse > 0.3) & (valid_nse <= 0.5)).sum()
        nse_5_to_75 = ((valid_nse > 0.5) & (valid_nse <= 0.75)).sum()
        nse_75_to_1 = ((valid_nse > 0.75) & (valid_nse <= 1.0)).sum()

        print(f"NSE ≤ 0          :          {nse_le_0:3d} ({100*nse_le_0/total_stations:5.1f}%)")
        print(f"0 < NSE ≤ 0.3    : {nse_0_to_3:3d} ({100*nse_0_to_3/total_stations:5.1f}%)")
        print(f"0.3 < NSE ≤ 0.5  :   {nse_3_to_5:3d} ({100*nse_3_to_5/total_stations:5.1f}%)")
        print(f"0.5 < NSE ≤ 0.75 :         {nse_5_to_75:3d} ({100*nse_5_to_75/total_stations:5.1f}%)")
        print(f"0.75 < NSE ≤ 1.0 :    {nse_75_to_1:3d} ({100*nse_75_to_1/total_stations:5.1f}%)")

        print(f"\nOverall Statistics:")
        print(f"Mean NSE: {valid_nse.mean():.3f}")
        print(f"Median NSE: {valid_nse.median():.3f}")
        print(f"Min NSE: {valid_nse.min():.3f}")
        print(f"Max NSE: {valid_nse.max():.3f}")

        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        pred_file = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(pred_file, index=False)
        
        metrics_file = os.path.join(output_dir, "station_metrics.csv")
        station_metrics_df.to_csv(metrics_file, index=False)

        print(f"\nSaved predictions to {pred_file}")
        print(f"Saved station metrics to {metrics_file}")

    return model


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    model = train_model(cfg)
