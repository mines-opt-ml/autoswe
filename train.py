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
    #station_normalizers = dataset.station_normalizers   # inputs
    swe_normalizers = dataset.swe_normalizers           # target SWE


    dataloader = SWEDataLoader(cfg)
    train_loader, val_loader, _, bt_info = dataloader.prepare()

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
                        date_str = dates[i][t]
                        if date_str not in backtrans_cache:
                            continue
                        try:
                            pred_val = preds[i, t].item()
                            # first undo SWE normalization (per station)
                            if swe_norm is not None:
                                pred_val = swe_norm.inverse_transform(
                                    pd.DataFrame([[pred_val]], columns=["SWE"])
                                )["SWE"].iloc[0]

                            # then apply spatial backtransform
                            pred_value = back_transform_scalar_with_weights(
                            pred_prime=pred_val,
                            station_idx=station_index[station],
                            date_str=date_str,
                            weights=weights,
                            backtrans_cache=backtrans_cache,
                            )

                            pred_value = max(0.0, pred_value)
                            actual_swe = swe_lookup.loc[(station, date_str)]

                            if epoch == 0 and len(val_preds) < 5:  # only print first 5 predictions in first epoch
                                print(f"[DEBUG] Station={station}, Date={date_str}, "
                                    f"Pred(raw)={preds[i, t].item():.3f}, "
                                    f"Pred(denorm)={pred_value:.3f}, "
                                    f"Actual={actual_swe:.3f}")

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

            # Per-epoch NSE distribution
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

    # ---------------- Final Metrics & Save ---------------- #
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions).sort_values(["station", "date"])
        station_metrics_df = compute_station_metrics(predictions_df).sort_values('nse', ascending=False)

        print(f"\nPer-Station Performance Summary:")
        print(f"{'Station':<12} {'NSE':<8} {'RMSE':<8} {'N':<6}")
        print("-" * 50)
        for _, row in station_metrics_df.iterrows():
            print(f"{row['station']:<12} {row['nse']:<8.3f} {row['rmse']:<8.2f} {row['n_predictions']:<6}")

        valid_nse = station_metrics_df['nse'].dropna()
        total_stations = len(valid_nse)
        print(f"\nPerformance Distribution across {total_stations} stations:")
        print(f"NSE ≤ 0          : { (valid_nse <= 0).sum():3d} ({100*(valid_nse <= 0).sum()/total_stations:5.1f}%)")
        print(f"0 < NSE ≤ 0.3    : { ((valid_nse > 0) & (valid_nse <= 0.3)).sum():3d} ({100*((valid_nse > 0) & (valid_nse <= 0.3)).sum()/total_stations:5.1f}%)")
        print(f"0.3 < NSE ≤ 0.5  : { ((valid_nse > 0.3) & (valid_nse <= 0.5)).sum():3d} ({100*((valid_nse > 0.3) & (valid_nse <= 0.5)).sum()/total_stations:5.1f}%)")
        print(f"0.5 < NSE ≤ 0.75 : { ((valid_nse > 0.5) & (valid_nse <= 0.75)).sum():3d} ({100*((valid_nse > 0.5) & (valid_nse <= 0.75)).sum()/total_stations:5.1f}%)")
        print(f"0.75 < NSE ≤ 1.0 : { ((valid_nse > 0.75) & (valid_nse <= 1.0)).sum():3d} ({100*((valid_nse > 0.75) & (valid_nse <= 1.0)).sum()/total_stations:5.1f}%)")

        print(f"\nOverall Statistics:")
        print(f"Mean NSE: {valid_nse.mean():.3f}")
        print(f"Median NSE: {valid_nse.median():.3f}")
        print(f"Min NSE: {valid_nse.min():.3f}")
        print(f"Max NSE: {valid_nse.max():.3f}")

        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
        station_metrics_df.to_csv(os.path.join(output_dir, "station_metrics.csv"), index=False)
        print(f"\nSaved predictions to results/predictions.csv")
        print(f"Saved station metrics to results/station_metrics.csv")

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
