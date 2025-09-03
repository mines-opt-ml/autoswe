import os
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_squared_error
import argparse
import glob
from dataloader import SWEDataLoader, SWEStationDataset
from modelzoo.LSTM import SWE_Net
from utils.backtransform import back_transform_scalar_with_weights
from utils.loss import get_loss_function


def compute_station_metrics(predictions_df: pd.DataFrame):
    """Compute NSE and RMSE per station given a predictions DataFrame."""
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
            'n_predictions': len(actual)
        })

    return pd.DataFrame(station_metrics)


def train_model(cfg: SimpleNamespace) -> SWE_Net:
    device = torch.device(cfg.device)
    
    dataset = SWEStationDataset(cfg)

    orig_swe = dataset.obs_swe.copy()
    orig_swe["Date"] = pd.to_datetime(orig_swe["Date"]).dt.strftime("%Y-%m-%d")
    swe_lookup = orig_swe.set_index(["Station", "Date"])["SWE"]
    swe_normalizers = dataset.swe_normalizers

    dataloader = SWEDataLoader(cfg)
    train_loader, val_loader, test_loader, _, bt_info = dataloader.prepare()

    station_index = bt_info["station_index"]
    weights = bt_info["weights"]
    backtrans_cache = bt_info["backtrans_cache"]

    raw_swe = dataset.obs_swe.copy()
    station_mean_swe = raw_swe.groupby("Station")["SWE"].mean()
    station_std_swe = raw_swe.groupby("Station")["SWE"].std()
    
    loss_fn = get_loss_function(cfg)

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

    all_predictions = []

    for epoch in range(cfg.n_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg.n_epochs}")
        print("-" * 50)

        # ---------------- Training ---------------- #
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            X = batch["dynamic forcing"].to(device)
            y = batch["swe"].to(device)   # already normalized by dataset
            stations = batch["station"]

            outputs = model(X, stations=stations)
            loss = loss_fn(outputs, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # ---------------- Validation ---------------- #
        model.eval()
        val_preds = []
        val_targets = []
        epoch_predictions = []

        with torch.no_grad():
            for batch in val_loader:
                X = batch["dynamic forcing"].to(device)
                mask = batch["mask"]
                stations = batch["station"]
                dates = batch["dates"]

                preds = model(X, stations=stations)

                for i in range(len(stations)):
                    station = stations[i]
                    valid_timesteps = mask[i].sum().item()
                    swe_norm = swe_normalizers.get(station, None)

                    for t in range(valid_timesteps):
                        date_str = pd.to_datetime(dates[i][t]).strftime("%Y-%m-%d")
                        if date_str not in backtrans_cache:
                            continue
                        try:
                            pred_prime = preds[i, t].item()
                            z_hat = back_transform_scalar_with_weights(
                                pred_prime=pred_prime,
                                station_idx=station_index[station],
                                date_str=date_str,
                                weights=weights,
                                backtrans_cache=backtrans_cache,
                            )
                            if swe_norm is not None:
                                pred_value = swe_norm.inverse_transform(
                                    pd.DataFrame([[z_hat]], columns=["SWE"])
                                )["SWE"].iloc[0]
                            else:
                                pred_value = z_hat

                            pred_value = max(0.0, pred_value)
                            actual_swe = swe_lookup.loc[(station, date_str)]

                            if epoch == cfg.n_epochs - 1:
                                all_predictions.append(
                                    {"station": station, "date": date_str,
                                     "predicted_swe": pred_value, "actual_swe": actual_swe}
                                )

                            val_preds.append(pred_value)
                            val_targets.append(actual_swe)
                            epoch_predictions.append({
                                "station": station,
                                "date": date_str,
                                "predicted_swe": pred_value,
                                "actual_swe": actual_swe
                            })

                        except Exception as e:
                            print(f"\nError at {station}, {date_str}: {str(e)}")
                            continue

        # ---------------- Metrics ---------------- #
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

            if epoch_predictions:
                predictions_df = pd.DataFrame(epoch_predictions)
                station_metrics_df = compute_station_metrics(predictions_df)
                valid_nse = station_metrics_df['nse'].dropna()

                nse_le_0 = (valid_nse <= 0).sum()
                nse_0_to_3 = ((valid_nse > 0) & (valid_nse <= 0.3)).sum()
                nse_3_to_5 = ((valid_nse > 0.3) & (valid_nse <= 0.5)).sum()
                nse_5_to_75 = ((valid_nse > 0.5) & (valid_nse <= 0.75)).sum()
                nse_75_to_1 = ((valid_nse > 0.75) & (valid_nse <= 1.0)).sum()

                print("\nEpoch NSE Distribution:")
                print(f"NSE ≤ 0          : {nse_le_0:3d}")
                print(f"0 < NSE ≤ 0.3    : {nse_0_to_3:3d}")
                print(f"0.3 < NSE ≤ 0.5  : {nse_3_to_5:3d}")
                print(f"0.5 < NSE ≤ 0.75 : {nse_5_to_75:3d}")
                print(f"0.75 < NSE ≤ 1.0 : {nse_75_to_1:3d}")

    # ---------------- Final Test Evaluation ---------------- #
    print("\nRunning on TEST set...")
    test_preds, test_targets, test_predictions = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            X = batch["dynamic forcing"].to(device)
            mask = batch["mask"]
            stations = batch["station"]
            dates = batch["dates"]

            preds = model(X, stations=stations)

            for i in range(len(stations)):
                station = stations[i]
                valid_timesteps = mask[i].sum().item()
                swe_norm = swe_normalizers.get(station, None)

                for t in range(valid_timesteps):
                    date_str = pd.to_datetime(dates[i][t]).strftime("%Y-%m-%d")
                    if date_str not in backtrans_cache:
                        continue
                    try:
                        pred_prime = preds[i, t].item()
                        z_hat = back_transform_scalar_with_weights(
                            pred_prime=pred_prime,
                            station_idx=station_index[station],
                            date_str=date_str,
                            weights=weights,
                            backtrans_cache=backtrans_cache,
                        )
                        if swe_norm is not None:
                            pred_value = swe_norm.inverse_transform(
                                pd.DataFrame([[z_hat]], columns=["SWE"])
                            )["SWE"].iloc[0]
                        else:
                            pred_value = z_hat
                        pred_value = max(0.0, pred_value)
                        actual_swe = swe_lookup.loc[(station, date_str)]

                        test_preds.append(pred_value)
                        test_targets.append(actual_swe)
                        test_predictions.append({
                            "station": station,
                            "date": date_str,
                            "predicted_swe": pred_value,
                            "actual_swe": actual_swe
                        })
                    except Exception as e:
                        print(f"\n[TEST] Error at {station}, {date_str}: {str(e)}")
                        continue

    if test_preds:
        rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        mean_observed = np.mean(test_targets)
        ss_res = np.sum((np.array(test_targets) - np.array(test_preds)) ** 2)
        ss_tot = np.sum((np.array(test_targets) - mean_observed) ** 2)
        nse = 1 - (ss_res / ss_tot)
        print(f"\nTEST Results — RMSE: {rmse:.4f}, NSE: {nse:.4f}")

        predictions_df = pd.DataFrame(test_predictions).sort_values(["station", "date"])
        station_metrics_df = compute_station_metrics(predictions_df).sort_values('nse', ascending=False)

        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        predictions_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
        station_metrics_df.to_csv(os.path.join(output_dir, "test_station_metrics.csv"), index=False)
        print(f"Saved test predictions to results/test_predictions.csv")
        print(f"Saved test station metrics to results/test_station_metrics.csv")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="Recompute decorrelation cache from scratch")
    args = parser.parse_args()

    if args.recompute:
        for f in glob.glob("decorrelated_dataset_*.pkl"):
            print(f"Removing cache file {f}")
            os.remove(f)

    with open("config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    model = train_model(cfg)
