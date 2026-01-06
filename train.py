import argparse
import os
import time
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_squared_error

from dataloader import SWEDataLoader
from modelzoo.HistoricalMean import HistoricalMean
from modelzoo.LSTM import SWE_Net
from utils.backtransform import back_transform_scalar_with_weights
from utils.metrics import masked_mse, masked_nse
from dataset import SWEStationDataset
from utils.snowyear_DOY_conversion import compute_day_of_snow_year

def compute_station_metrics(predictions_df: pd.DataFrame):
    """
    Compute NSE and RMSE per station. 
    Expects: ['station','actual_swe','predicted_swe'].
    """
    station_metrics = []
    for station in predictions_df["station"].unique():
        station_data = predictions_df[predictions_df["station"] == station]
        actual = station_data["actual_swe"].values
        predicted = station_data["predicted_swe"].values
        mean_observed = np.mean(actual)
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((np.array(actual) - mean_observed) ** 2)
        nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        station_metrics.append({"station": station, "nse": nse, "rmse": rmse, "n_predictions": len(actual)})
    return pd.DataFrame(station_metrics)


def build_backtrans_cache_normalized_from_obs(obs_swe: pd.DataFrame, swe_normalizers: dict, station_index: dict, weights) -> dict:
    """Build per-date offsets using normalized SWE so the Heaton back-transform returns normalized predictions."""
    df = obs_swe.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Station"] = df["Station"].str.lower()
    inv = {idx: st for st, idx in station_index.items()}
    station_order = [inv[i] for i in range(len(inv))]
    means = []
    stds = []
    for st in station_order:
        stats = swe_normalizers[st].stats["SWE"]
        means.append(float(stats["mean"]))
        stds.append(float(stats["std"]))
    means = np.array(means, dtype=np.float32)
    stds = np.array(stds, dtype=np.float32)
    cache = {}
    for dt in sorted(df["Date"].unique()):
        day = df[df["Date"] == dt].set_index("Station").reindex(station_order)
        y = day["SWE"].to_numpy(dtype=np.float32)
        y_norm = (y - means) / (stds + 1e-6)
        y_norm = np.nan_to_num(y_norm, nan=0.0)
        offsets = (weights.A * y_norm[weights.nn_index]).sum(axis=1).astype(np.float32)
        cache[pd.Timestamp(dt).strftime("%Y-%m-%d")] = offsets
    return cache


def build_doy_climatology(obs_swe: pd.DataFrame, train_start: int, train_end: int) -> Tuple[pd.Series, pd.Series]:
    """
    Build per-station, per-DOY climatology using train years only.
    """
    df = obs_swe.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # The line below assumes the train years are contiguous. Consider changing to something like in(train_years) for robustness.
    train_years = list(range(train_start, train_end + 1))
    mask = df["Date"].dt.year.isin(train_years)
    #mask = (df["Date"].dt.year >= train_start) & (df["Date"].dt.year <= train_end)
    train_df = df.loc[mask].copy()
    train_df["DOY"] = train_df["Date"].dt.dayofyear
    climo = train_df.groupby(["Station", "DOY"])["SWE"].mean()
    station_mean = train_df.groupby("Station")["SWE"].mean()
    return climo, station_mean

def run_validation(*, model: torch.nn.Module, val_loader, hist_mean_model,
    swe_normalizers: dict, backtrans_cache: dict, station_index: dict, weights,
    swe_lookup: pd.Series, cfg, avg_train_loss: float, epoch_start_time: float,
):
    """
    Standalone validation function.
    """
    device = next(model.parameters()).device
    model.eval()
    val_preds, val_targets, epoch_predictions = [], [], []
    val_baseline = []

    with torch.no_grad():
        for batch in val_loader:
            X = batch["dynamic forcing"].to(device)
            stations = batch["station"]
            dates = batch["dates"]
            preds = model(X, stations=stations)
            preds_base_swe = hist_mean_model(X, stations=stations, dates=dates)

            if getattr(cfg, "anomaly_target", False):
                if "swe_climo" not in batch:
                    raise RuntimeError("anomaly_target=True but 'swe_climo' missing from batch.")
                climo = batch["swe_climo"].to(device)
            else:
                climo = None

            mask = batch["mask"]
            for i in range(len(stations)):
                station = stations[i]
                valid_timesteps = int(mask[i].sum().item())
                swe_norm = swe_normalizers.get(station, None)
                for t in range(valid_timesteps):
                    date_str = pd.to_datetime(dates[i][t]).strftime("%Y-%m-%d")
                    if date_str not in backtrans_cache:
                        continue
                    pred_prime = (preds[i, t] + climo[i, t]).item() if climo is not None else preds[i, t].item()
                    z_hat = back_transform_scalar_with_weights(
                        pred_prime=pred_prime,
                        station_idx=station_index[station],
                        date_str=date_str,
                        weights=weights,
                        backtrans_cache=backtrans_cache,
                    )
                    if swe_norm is not None:
                        pred_value = swe_norm.inverse_transform(pd.DataFrame([[z_hat]], columns=["SWE"]))["SWE"].iloc[0]
                    else:
                        pred_value = z_hat
                    pred_value = max(0.0, pred_value)
                    actual_swe = swe_lookup.loc[(station, date_str)]
                    base_val = max(0.0, float(preds_base_swe[i, t].item()))
                    val_baseline.append(base_val)
                    val_preds.append(pred_value)
                    val_targets.append(actual_swe)
                    epoch_predictions.append(
                        {
                            "station": station,
                            "date": date_str,
                            "predicted_swe": pred_value,
                            "baseline_swe": base_val,
                            "actual_swe": actual_swe,
                        }
                    )

    metrics = {}
    if val_preds:
        rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        mean_observed = np.mean(val_targets)
        ss_res = np.sum((np.array(val_targets) - np.array(val_preds)) ** 2)
        ss_tot = np.sum((np.array(val_targets) - mean_observed) ** 2)
        nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        epoch_time = time.time() - epoch_start_time
        print(f"Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val RMSE: {rmse:.4f} | Val NSE: {nse:.4f}")

        if val_baseline:
            base_rmse = np.sqrt(mean_squared_error(val_targets, val_baseline))
            base_ss_res = np.sum((np.array(val_targets) - np.array(val_baseline)) ** 2)
            base_nse = 1 - (base_ss_res / ss_tot) if ss_tot > 0 else np.nan
            skill_rmse = 1.0 - (rmse / base_rmse) if base_rmse > 0 else np.nan
            print(
                f"Baseline (Climatology) → RMSE: {base_rmse:.4f} | NSE: {base_nse:.4f} | RMSE Skill vs Clim: {skill_rmse:.4f}"
            )

        station_metrics_df = compute_station_metrics(pd.DataFrame(epoch_predictions))
        valid_nse = station_metrics_df["nse"].dropna()
        nse_le_0 = (valid_nse <= 0).sum()
        nse_0_to_3 = ((valid_nse > 0) & (valid_nse <= 0.3)).sum()
        nse_3_to_5 = ((valid_nse > 0.3) & (valid_nse <= 0.5)).sum()
        nse_5_to_75 = ((valid_nse > 0.5) & (valid_nse <= 0.75)).sum()
        nse_75_to_1 = ((valid_nse > 0.75) & (valid_nse <= 1.0)).sum()
        print("\nEpoch NSE Distribution (Model):")
        print(f"NSE ≤ 0          : {nse_le_0:3d}")
        print(f"0 < NSE ≤ 0.3    : {nse_0_to_3:3d}")
        print(f"0.3 < NSE ≤ 0.5  : {nse_3_to_5:3d}")
        print(f"0.5 < NSE ≤ 0.75 : {nse_5_to_75:3d}")
        print(f"0.75 < NSE ≤ 1.0 : {nse_75_to_1:3d}")

        metrics.update(
            {
                "val_rmse": float(rmse),
                "val_nse": float(nse),
                "val_ss_tot": float(ss_tot),
                "station_metrics_df": station_metrics_df,
            }
        )

    return metrics

def run_test(*, model: torch.nn.Module, test_loader, hist_mean_model,
    swe_normalizers: dict, backtrans_cache: dict, station_index: dict, weights, 
    swe_lookup: pd.Series, best_model_state, device: torch.device, best_epoch: int, cfg,
):
    """
    Standalone test function.
    """
    print("\nLoading best model checkpoint for TEST evaluation...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif os.path.exists("results/best_model.pt"):
        model.load_state_dict(torch.load("results/best_model.pt", map_location=device))
    else:
        print("No best model checkpoint found; using last epoch model.")

    print("\nRunning on TEST set...")
    test_preds, test_targets, test_predictions = [], [], []
    test_baseline = []
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            X = batch["dynamic forcing"].to(device)
            mask = batch["mask"]
            stations = batch["station"]
            dates = batch["dates"]
            preds = model(X, stations=stations)
            preds_base_swe = hist_mean_model(X, stations=stations, dates=dates)

            if getattr(cfg, "anomaly_target", False):
                if "swe_climo" not in batch:
                    raise RuntimeError("anomaly_target=True but 'swe_climo' missing from batch.")
                climo = batch["swe_climo"].to(device)
            else:
                climo = None

            for i in range(len(stations)):
                station = stations[i]
                valid_timesteps = int(mask[i].sum().item())
                swe_norm = swe_normalizers.get(station, None)
                for t in range(valid_timesteps):
                    date_str = pd.to_datetime(dates[i][t]).strftime("%Y-%m-%d")
                    if date_str not in backtrans_cache:
                        continue
                    pred_prime = (preds[i, t] + climo[i, t]).item() if climo is not None else preds[i, t].item()
                    z_hat = back_transform_scalar_with_weights(
                        pred_prime=pred_prime,
                        station_idx=station_index[station],
                        date_str=date_str,
                        weights=weights,
                        backtrans_cache=backtrans_cache,
                    )
                    if swe_norm is not None:
                        pred_value = swe_norm.inverse_transform(pd.DataFrame([[z_hat]], columns=["SWE"]))["SWE"].iloc[0]
                    else:
                        pred_value = z_hat
                    pred_value = max(0.0, pred_value)
                    actual_swe = swe_lookup.loc[(station, date_str)]
                    base_val = max(0.0, float(preds_base_swe[i, t].item()))
                    test_baseline.append(base_val)
                    test_preds.append(pred_value)
                    test_targets.append(actual_swe)
                    test_predictions.append(
                        {
                            "station": station,
                            "date": date_str,
                            "predicted_swe": pred_value,
                            "baseline_swe": base_val,
                            "actual_swe": actual_swe,
                        }
                    )

    if test_preds:
        rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
        mean_observed = np.mean(test_targets)
        ss_res = np.sum((np.array(test_targets) - np.array(test_preds)) ** 2)
        ss_tot = np.sum((np.array(test_targets) - mean_observed) ** 2)
        nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        print(f"\nTEST Results (Best Epoch {best_epoch})")
        print(f"TEST RMSE: {rmse:.4f}")
        print(f"TEST NSE : {nse:.4f}")
        print(f"TEST Predictions: {len(test_preds)}")

        station_metrics_df = compute_station_metrics(pd.DataFrame(test_predictions))
        valid_nse = station_metrics_df["nse"].dropna()
        print("\nStation-level NSE distribution (TEST, Model):")
        print(f"NSE ≤ 0          : {(valid_nse <= 0).sum():3d}")
        print(f"0 < NSE ≤ 0.3    : {((valid_nse > 0) & (valid_nse <= 0.3)).sum():3d}")
        print(f"0.3 < NSE ≤ 0.5  : {((valid_nse > 0.3) & (valid_nse <= 0.5)).sum():3d}")
        print(f"0.5 < NSE ≤ 0.75 : {((valid_nse > 0.5) & (valid_nse <= 0.75)).sum():3d}")
        print(f"0.75 < NSE ≤ 1.0 : {((valid_nse > 0.75) & (valid_nse <= 1.0)).sum():3d}")

        if test_baseline:
            base_rmse = np.sqrt(mean_squared_error(test_targets, test_baseline))
            base_ss_res = np.sum((np.array(test_targets) - np.array(test_baseline)) ** 2)
            base_nse = 1 - (base_ss_res / ss_tot) if ss_tot > 0 else np.nan
            skill_rmse = 1.0 - (rmse / base_rmse) if base_rmse > 0 else np.nan
            print("\nBaseline (Climatology) on TEST")
            print(f"TEST Baseline RMSE: {base_rmse:.4f}")
            print(f"TEST Baseline NSE : {base_nse:.4f}")
            print(f"TEST RMSE Skill vs Clim: {skill_rmse:.4f}")

            baseline_metrics_df = compute_station_metrics(
                pd.DataFrame(test_predictions)[["station", "baseline_swe", "actual_swe"]].rename(
                    columns={"baseline_swe": "predicted_swe"}
                )
            )
            valid_nse_base = baseline_metrics_df["nse"].dropna()
            print("\nStation-level NSE distribution (TEST, Baseline):")
            print(f"NSE ≤ 0          : {(valid_nse_base <= 0).sum():3d}")
            print(f"0 < NSE ≤ 0.3    : {((valid_nse_base > 0) & (valid_nse_base <= 0.3)).sum():3d}")
            print(f"0.3 < NSE ≤ 0.5  : {((valid_nse_base > 0.3) & (valid_nse_base <= 0.5)).sum():3d}")
            print(f"0.5 < NSE ≤ 0.75 : {((valid_nse_base > 0.5) & (valid_nse_base <= 0.75)).sum():3d}")
            print(f"0.75 < NSE ≤ 1.0 : {((valid_nse_base > 0.75) & (valid_nse_base <= 1.0)).sum():3d}")

        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(test_predictions).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
        station_metrics_df.to_csv(os.path.join(output_dir, "test_station_metrics.csv"), index=False)
        preds_with_baseline = pd.DataFrame(test_predictions)
        preds_with_baseline.to_csv(os.path.join(output_dir, "test_predictions_with_baseline.csv"), index=False)
        baseline_df = preds_with_baseline[["station", "date", "baseline_swe", "actual_swe"]].rename(
            columns={"baseline_swe": "predicted_swe"}
        )
        station_metrics_baseline = compute_station_metrics(baseline_df)[["station", "nse", "rmse", "n_predictions"]]
        station_metrics_baseline.to_csv(os.path.join(output_dir, "test_station_metrics_baseline.csv"), index=False)
        print("Saved test predictions to results/test_predictions.csv")
        print("Saved test station metrics to results/test_station_metrics.csv")
        print("Saved test predictions (+ baseline) to results/test_predictions_with_baseline.csv")
        print("Saved test station metrics (baseline) to results/test_station_metrics_baseline.csv")

    return {
        "test_rmse": float(rmse) if test_preds else None,
        "test_nse": float(nse) if test_preds else None,
        "num_predictions": int(len(test_preds)),
        "best_epoch": int(best_epoch),
    }

def train_model(cfg: SimpleNamespace):
    device = torch.device(cfg.device)

    dataset = SWEStationDataset(cfg)
    orig_swe = dataset.obs_swe.copy()
    orig_swe["Date"] = pd.to_datetime(orig_swe["Date"]).dt.strftime("%Y-%m-%d")
    swe_lookup = orig_swe.set_index(["Station", "Date"])["SWE"]
    swe_normalizers = dataset.swe_normalizers

    climo, station_mean = build_doy_climatology(
        obs_swe=dataset.obs_swe, train_start=cfg.train_start_year, train_end=cfg.train_end_year
    )

    dataloader = SWEDataLoader(cfg)
    train_loader, val_loader, test_loader, _, bt_info = dataloader.prepare()
    station_index = bt_info["station_index"]
    weights = bt_info["weights"]

    backtrans_cache = build_backtrans_cache_normalized_from_obs(
        obs_swe=dataset.obs_swe, swe_normalizers=swe_normalizers, station_index=station_index, weights=weights
    )

    target_swe = dataset.obs_swe.copy()
    target_swe["Date"] = pd.to_datetime(target_swe["Date"])
    m = (target_swe["Date"].dt.year >= cfg.train_start_year) & (target_swe["Date"].dt.year <= cfg.train_end_year)
    tr = target_swe.loc[m]
    station_mean_swe = tr.groupby("Station")["SWE"].mean()
    station_std_swe = tr.groupby("Station")["SWE"].std()
    station_stats_dict = {}
    for station in station_mean_swe.index:
        station_stats_dict[station] = (float(station_mean_swe[station]), float(station_std_swe[station]))

    sample0 = next(iter(train_loader))
    cfg.input_size = sample0["dynamic forcing"].shape[-1]

    model = SWE_Net(cfg, station_stats=station_stats_dict).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=8, factor=0.7)
    hist_mean_model = HistoricalMean(climo_lookup=climo.to_dict(), station_mean=station_mean.to_dict()).to(device)
    hist_mean_model.eval()

    best_val_nse = float("-inf")
    best_epoch = -1
    best_model_state = None

    for epoch in range(cfg.n_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg.n_epochs}")
        print("-" * 50)

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch["dynamic forcing"].to(device)
            y = batch["swe"].to(device)
            stations = batch["station"]

            outputs = model(X, stations=stations)

            if getattr(cfg, "anomaly_target", False):
                if "swe_climo" not in batch:
                    raise RuntimeError("anomaly_target=True but 'swe_climo' missing from batch.")
                climo_batch = batch["swe_climo"].to(device)
                y_target = y - climo_batch
            else:
                y_target = y

            mask = batch["mask"].to(outputs.device).float()

            if isinstance(cfg.loss, str):
                if cfg.loss.upper() == "MSE":
                    loss = masked_mse(outputs, y, mask)
                elif cfg.loss.upper() == "NSE":
                    loss = masked_nse(outputs, y, mask)
                else:
                    raise ValueError(f"Unknown loss: {cfg.loss}")
            elif isinstance(cfg.loss, (list, tuple)):
                weights_ = cfg.loss_weights
                terms = []
                for w, name in zip(weights_, cfg.loss):
                    name = name.upper()
                    if name == "MSE":
                        terms.append(w * masked_mse(outputs, y, mask))
                    elif name == "NSE":
                        terms.append(w * masked_nse(outputs, y, mask))
                    else:
                        raise ValueError(f"Unknown loss: {name}")
                loss = sum(terms)
            else:
                raise ValueError("cfg.loss must be str or list")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        metrics = run_validation(
            model=model,
            val_loader=val_loader,
            hist_mean_model=hist_mean_model,
            swe_normalizers=swe_normalizers,
            backtrans_cache=backtrans_cache,
            station_index=station_index,
            weights=weights,
            swe_lookup=swe_lookup,
            cfg=cfg,
            avg_train_loss=avg_train_loss,
            epoch_start_time=epoch_start,
        )

        if metrics:
            nse = metrics["val_nse"]
            scheduler.step(nse)
            if nse > best_val_nse:
                best_val_nse = nse
                best_epoch = epoch + 1
                best_model_state = model.state_dict()
                os.makedirs("results", exist_ok=True)
                torch.save(best_model_state, "results/best_model.pt")
                print(f"*** New best model saved (Epoch {best_epoch}, Val NSE = {best_val_nse:.4f}) ***")

    _ = run_test(
        model=model,
        test_loader=test_loader,
        hist_mean_model=hist_mean_model,
        swe_normalizers=swe_normalizers,
        backtrans_cache=backtrans_cache,
        station_index=station_index,
        weights=weights,
        swe_lookup=swe_lookup,
        best_model_state=best_model_state,
        device=device,
        best_epoch=best_epoch,
        cfg = cfg,
    )

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open("config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    model = train_model(cfg)