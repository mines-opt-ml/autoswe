import argparse
import os
import random
import time
from types import SimpleNamespace
from typing import Mapping, Tuple

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


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_station_metrics(predictions_df: pd.DataFrame):
    """
    Compute NSE and RMSE per station.
    Expects columns: ['station', 'actual_swe', 'predicted_swe'].
    """
    station_metrics = []
    for station in predictions_df["station"].unique():
        station_data = predictions_df[predictions_df["station"] == station]
        actual = station_data["actual_swe"].values
        predicted = station_data["predicted_swe"].values

        mean_observed = np.mean(actual)
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - mean_observed) ** 2)
        nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        station_metrics.append(
            {
                "station": station,
                "nse": nse,
                "rmse": rmse,
                "n_predictions": len(actual),
            }
        )

    return pd.DataFrame(station_metrics)


def build_backtrans_cache_normalized_from_obs(
    obs_swe: pd.DataFrame,
    swe_normalizers: dict,
    station_index: dict,
    weights,
) -> dict:
    """
    Build per-date offsets using normalized SWE so the Heaton back-transform
    returns normalized predictions.
    """
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
    train_years = list(range(train_start, train_end + 1))
    mask = df["Date"].dt.year.isin(train_years)
    train_df = df.loc[mask].copy()

    train_df["DOY"] = train_df["Date"].dt.dayofyear
    climo = train_df.groupby(["Station", "DOY"])["SWE"].mean()
    station_mean = train_df.groupby("Station")["SWE"].mean()

    return climo, station_mean


def build_persistence_df(predictions_df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Build a simple persistence baseline:
        predicted_swe_t = actual_swe_(t-1)
    computed within each station after sorting by date.
    """
    df = predictions_df[["station", "date", "actual_swe"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station", "date"]).copy()

    df["predicted_swe"] = df.groupby("station")["actual_swe"].shift(1)
    df["predicted_swe"] = df["predicted_swe"].fillna(fill_value)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["station", "date", "predicted_swe", "actual_swe"]]


def print_nse_distribution(metrics_df: pd.DataFrame, title: str):
    valid_nse = metrics_df["nse"].dropna()
    print(f"\n{title}")
    print(f"NSE ≤ 0          : {(valid_nse <= 0).sum():3d}")
    print(f"0 < NSE ≤ 0.3    : {((valid_nse > 0) & (valid_nse <= 0.3)).sum():3d}")
    print(f"0.3 < NSE ≤ 0.5  : {((valid_nse > 0.3) & (valid_nse <= 0.5)).sum():3d}")
    print(f"0.5 < NSE ≤ 0.75 : {((valid_nse > 0.5) & (valid_nse <= 0.75)).sum():3d}")
    print(f"0.75 < NSE ≤ 1.0 : {((valid_nse > 0.75) & (valid_nse <= 1.0)).sum():3d}")


def compute_global_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mean_observed = np.mean(actual)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - mean_observed) ** 2)
    nse = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return rmse, nse, ss_tot


def collect_predictions(
    *,
    model: torch.nn.Module,
    loader,
    hist_mean_model,
    swe_normalizers: dict,
    backtrans_cache: dict,
    station_index: dict,
    weights,
    swe_lookup: pd.Series,
    cfg,
):
    """
    Run a loader and return row-level predictions on the original SWE scale.
    """
    device = next(model.parameters()).device
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in loader:
            X = batch["dynamic forcing"].to(device)
            mask = batch["mask"]
            stations = batch["station"]
            dates = batch["dates"]

            preds = model(X, stations=stations)
            preds_base_clim = hist_mean_model(X, stations=stations, dates=dates)

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
                        pred_value = swe_norm.inverse_transform(
                            pd.DataFrame([[z_hat]], columns=["SWE"])
                        )["SWE"].iloc[0]
                    else:
                        pred_value = z_hat

                    actual_swe = swe_lookup.loc[(station, date_str)]
                    base_clim_val = max(0.0, float(preds_base_clim[i, t].item()))

                    predictions.append(
                        {
                            "station": station,
                            "date": date_str,
                            "predicted_swe": pred_value,
                            "baseline_swe": base_clim_val,
                            "actual_swe": actual_swe,
                        }
                    )

    return pd.DataFrame(predictions)


def conformal_quantile(calibration_df: pd.DataFrame, alpha: float) -> float:
    """
    Split-conformal quantile for absolute residual intervals.
    """
    if calibration_df.empty:
        raise ValueError("Cannot compute conformal quantile from an empty calibration set.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"conformal_alpha must be between 0 and 1, got {alpha}.")

    residuals = np.abs(
        calibration_df["actual_swe"].to_numpy(dtype=float)
        - calibration_df["predicted_swe"].to_numpy(dtype=float)
    )
    n = len(residuals)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(residuals, q_level, method="higher"))


def conformal_quantiles_by_station(calibration_df: pd.DataFrame, alpha: float) -> dict:
    """
    Split-conformal absolute residual quantiles computed independently per station.
    """
    if "station" not in calibration_df:
        raise ValueError("Calibration dataframe must include a 'station' column.")

    return {
        station: conformal_quantile(station_df, alpha)
        for station, station_df in calibration_df.groupby("station")
    }


def add_conformal_intervals(
    predictions_df: pd.DataFrame,
    q_hat_by_station: Mapping[str, float],
    fallback_q_hat: float = None,
) -> pd.DataFrame:
    """
    Attach station-specific nonnegative lower and upper split-conformal SWE intervals.
    """
    df = predictions_df.copy()
    df["conformal_q_hat"] = df["station"].map(q_hat_by_station)
    if fallback_q_hat is not None:
        df["conformal_q_hat"] = df["conformal_q_hat"].fillna(fallback_q_hat)
    if df["conformal_q_hat"].isna().any():
        missing_stations = sorted(df.loc[df["conformal_q_hat"].isna(), "station"].unique())
        raise ValueError(
            "Missing conformal quantile for station(s): "
            + ", ".join(str(station) for station in missing_stations)
        )

    q_hat = df["conformal_q_hat"].to_numpy(dtype=float)
    df["conformal_lower_swe"] = np.maximum(0.0, df["predicted_swe"] - q_hat)
    df["conformal_upper_swe"] = df["predicted_swe"] + q_hat
    covered = (
        (df["actual_swe"] >= df["conformal_lower_swe"])
        & (df["actual_swe"] <= df["conformal_upper_swe"])
    )
    df["conformal_covered"] = covered
    return df


def run_validation(
    *,
    model: torch.nn.Module,
    val_loader,
    hist_mean_model,
    swe_normalizers: dict,
    backtrans_cache: dict,
    station_index: dict,
    weights,
    swe_lookup: pd.Series,
    cfg,
    avg_train_loss: float,
    epoch_start_time: float,
):
    """
    Standalone validation function.
    """
    epoch_predictions_df = collect_predictions(
        model=model,
        loader=val_loader,
        hist_mean_model=hist_mean_model,
        swe_normalizers=swe_normalizers,
        backtrans_cache=backtrans_cache,
        station_index=station_index,
        weights=weights,
        swe_lookup=swe_lookup,
        cfg=cfg,
    )

    metrics = {}
    if not epoch_predictions_df.empty:
        val_preds = epoch_predictions_df["predicted_swe"].values
        val_targets = epoch_predictions_df["actual_swe"].values
        val_baseline_clim = epoch_predictions_df["baseline_swe"].values

        rmse, nse, ss_tot = compute_global_metrics(val_targets, val_preds)
        epoch_time = time.time() - epoch_start_time

        print(f"Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val RMSE: {rmse:.4f} | Val NSE: {nse:.4f}")

        if len(val_baseline_clim) > 0:
            base_rmse, base_nse, _ = compute_global_metrics(val_targets, val_baseline_clim)
            skill_rmse = 1.0 - (rmse / base_rmse) if base_rmse > 0 else np.nan
            print(
                f"Baseline (Climatology) → RMSE: {base_rmse:.4f} | NSE: {base_nse:.4f} | RMSE Skill vs Clim: {skill_rmse:.4f}"
            )

        station_metrics_df = compute_station_metrics(epoch_predictions_df)
        print_nse_distribution(station_metrics_df, "Epoch NSE Distribution (Model):")

        persistence_df = build_persistence_df(epoch_predictions_df, fill_value=0.0)
        persist_rmse, persist_nse, _ = compute_global_metrics(
            persistence_df["actual_swe"].values,
            persistence_df["predicted_swe"].values,
        )
        persist_skill = 1.0 - (rmse / persist_rmse) if persist_rmse > 0 else np.nan
        #print(
        #    f"Baseline (Persistence) → RMSE: {persist_rmse:.4f} | NSE: {persist_nse:.4f} | RMSE Skill vs Persist: {persist_skill:.4f}"
        #)

        metrics.update(
            {
                "val_rmse": float(rmse),
                "val_nse": float(nse),
                "val_ss_tot": float(ss_tot),
                "station_metrics_df": station_metrics_df,
            }
        )

    return metrics


def run_test(
    *,
    model: torch.nn.Module,
    test_loader,
    hist_mean_model,
    swe_normalizers: dict,
    backtrans_cache: dict,
    station_index: dict,
    weights,
    swe_lookup: pd.Series,
    best_model_state,
    device: torch.device,
    best_epoch: int,
    cfg,
    conformal_q_hat_by_station: Mapping[str, float] = None,
    conformal_global_q_hat: float = None,
    conformal_alpha: float = None,
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
    test_preds = []
    test_predictions_df = collect_predictions(
        model=model,
        loader=test_loader,
        hist_mean_model=hist_mean_model,
        swe_normalizers=swe_normalizers,
        backtrans_cache=backtrans_cache,
        station_index=station_index,
        weights=weights,
        swe_lookup=swe_lookup,
        cfg=cfg,
    )

    num_predictions = int(len(test_predictions_df))
    if num_predictions > 0:
        test_preds = test_predictions_df["predicted_swe"].values
        test_targets = test_predictions_df["actual_swe"].values
        test_baseline_clim = test_predictions_df["baseline_swe"].values
        rmse, nse, _ = compute_global_metrics(test_targets, test_preds)

        print(f"\nTEST Results (Best Epoch {best_epoch})")
        print(f"TEST RMSE: {rmse:.4f}")
        print(f"TEST NSE : {nse:.4f}")
        print(f"TEST Predictions: {len(test_preds)}")

        if conformal_q_hat_by_station is not None:
            test_predictions_df = add_conformal_intervals(
                test_predictions_df,
                conformal_q_hat_by_station,
                fallback_q_hat=conformal_global_q_hat,
            )
            coverage = float(test_predictions_df["conformal_covered"].mean())
            interval_width = float(
                (test_predictions_df["conformal_upper_swe"] - test_predictions_df["conformal_lower_swe"]).mean()
            )
            station_coverage = test_predictions_df.groupby("station")["conformal_covered"].mean()
            q_hats = np.array(list(conformal_q_hat_by_station.values()), dtype=float)
            target_coverage = 1.0 - conformal_alpha if conformal_alpha is not None else np.nan
            print("\nStation-specific split-conformal intervals on TEST")
            print(f"Target coverage: {target_coverage:.3f}")
            print(f"Observed coverage: {coverage:.3f}")
            print(f"Median station coverage: {station_coverage.median():.3f}")
            print(f"Station q_hat median: {np.median(q_hats):.4f}")
            print(f"Station q_hat range: {np.min(q_hats):.4f} to {np.max(q_hats):.4f}")
            print(f"Mean interval width: {interval_width:.4f}")

        station_metrics_df = compute_station_metrics(test_predictions_df)
        print_nse_distribution(station_metrics_df, "Station-level NSE distribution (TEST, Model):")

        if len(test_baseline_clim) > 0:
            base_rmse, base_nse, _ = compute_global_metrics(test_targets, test_baseline_clim)
            skill_rmse = 1.0 - (rmse / base_rmse) if base_rmse > 0 else np.nan

            print("\nBaseline (Climatology) on TEST")
            print(f"TEST Baseline RMSE: {base_rmse:.4f}")
            print(f"TEST Baseline NSE : {base_nse:.4f}")
            print(f"TEST RMSE Skill vs Clim: {skill_rmse:.4f}")

            baseline_df = test_predictions_df[["station", "date", "baseline_swe", "actual_swe"]].rename(
                columns={"baseline_swe": "predicted_swe"}
            )
            baseline_metrics_df = compute_station_metrics(baseline_df)
            print_nse_distribution(
                baseline_metrics_df,
                "Station-level NSE distribution (TEST, Baseline / Climatology):",
            )

        persistence_df = build_persistence_df(test_predictions_df, fill_value=0.0)
        persist_rmse, persist_nse, _ = compute_global_metrics(
            persistence_df["actual_swe"].values,
            persistence_df["predicted_swe"].values,
        )
        persist_skill = 1.0 - (rmse / persist_rmse) if persist_rmse > 0 else np.nan

        #print("\nBaseline (Persistence) on TEST")
        #print(f"TEST Persistence RMSE: {persist_rmse:.4f}")
        #print(f"TEST Persistence NSE : {persist_nse:.4f}")
        #print(f"TEST RMSE Skill vs Persist: {persist_skill:.4f}")

        persistence_metrics_df = compute_station_metrics(persistence_df)
        print_nse_distribution(
            persistence_metrics_df,
            "Station-level NSE distribution (TEST, Baseline / Persistence):",
        )

        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)

        test_predictions_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
        station_metrics_df.to_csv(os.path.join(output_dir, "test_station_metrics.csv"), index=False)

        if conformal_q_hat_by_station is not None:
            conformal_q_hat_df = pd.DataFrame(
                [
                    {"station": station, "conformal_q_hat": q_hat}
                    for station, q_hat in sorted(conformal_q_hat_by_station.items())
                ]
            )
            conformal_q_hat_df.to_csv(
                os.path.join(output_dir, "conformal_q_hat_by_station.csv"),
                index=False,
            )

        preds_with_baseline = test_predictions_df.copy()
        preds_with_baseline = preds_with_baseline.merge(
            persistence_df[["station", "date", "predicted_swe"]].rename(
                columns={"predicted_swe": "persistence_swe"}
            ),
            on=["station", "date"],
            how="left",
        )
        preds_with_baseline.to_csv(
            os.path.join(output_dir, "test_predictions_with_baseline.csv"),
            index=False,
        )

        baseline_df = preds_with_baseline[["station", "date", "baseline_swe", "actual_swe"]].rename(
            columns={"baseline_swe": "predicted_swe"}
        )
        station_metrics_baseline = compute_station_metrics(
            baseline_df
        )[["station", "nse", "rmse", "n_predictions"]]
        station_metrics_baseline.to_csv(
            os.path.join(output_dir, "test_station_metrics_baseline.csv"),
            index=False,
        )

        persistence_df.to_csv(
            os.path.join(output_dir, "test_predictions_persistence.csv"),
            index=False,
        )
        persistence_metrics_df[["station", "nse", "rmse", "n_predictions"]].to_csv(
            os.path.join(output_dir, "test_station_metrics_persistence.csv"),
            index=False,
        )

        print("Saved test predictions to results/test_predictions.csv")
        print("Saved test station metrics to results/test_station_metrics.csv")
        if conformal_q_hat_by_station is not None:
            print("Saved station-specific conformal q_hats to results/conformal_q_hat_by_station.csv")
        print("Saved test predictions (+ baseline) to results/test_predictions_with_baseline.csv")
        print("Saved test station metrics (baseline) to results/test_station_metrics_baseline.csv")
        print("Saved test predictions (persistence) to results/test_predictions_persistence.csv")
        print("Saved test station metrics (persistence) to results/test_station_metrics_persistence.csv")

    return {
        "test_rmse": float(rmse) if num_predictions > 0 else None,
        "test_nse": float(nse) if num_predictions > 0 else None,
        "num_predictions": num_predictions,
        "best_epoch": int(best_epoch),
        "conformal_q_hat_by_station": (
            dict(conformal_q_hat_by_station)
            if conformal_q_hat_by_station is not None
            else None
        ),
        "conformal_global_q_hat": float(conformal_global_q_hat) if conformal_global_q_hat is not None else None,
    }


def train_model(cfg: SimpleNamespace):
    if hasattr(cfg, "seed"):
        set_seed(cfg.seed, getattr(cfg, "deterministic", False))

    total_start_time = time.time()
    device = torch.device(cfg.device)

    dataset = SWEStationDataset(cfg)
    orig_swe = dataset.obs_swe.copy()
    orig_swe["Date"] = pd.to_datetime(orig_swe["Date"]).dt.strftime("%Y-%m-%d")
    swe_lookup = orig_swe.set_index(["Station", "Date"])["SWE"]
    swe_normalizers = dataset.swe_normalizers

    climo, station_mean = build_doy_climatology(
        obs_swe=dataset.obs_swe,
        train_start=cfg.train_start_year,
        train_end=cfg.train_end_year,
    )

    dataloader = SWEDataLoader(cfg)
    train_loader, val_loader, test_loader, _, bt_info = dataloader.prepare()
    station_index = bt_info["station_index"]
    weights = bt_info["weights"]

    backtrans_cache = build_backtrans_cache_normalized_from_obs(
        obs_swe=dataset.obs_swe,
        swe_normalizers=swe_normalizers,
        station_index=station_index,
        weights=weights,
    )

    target_swe = dataset.obs_swe.copy()
    target_swe["Date"] = pd.to_datetime(target_swe["Date"])
    m = (target_swe["Date"].dt.year >= cfg.train_start_year) & (target_swe["Date"].dt.year <= cfg.train_end_year)
    tr = target_swe.loc[m]

    station_mean_swe = tr.groupby("Station")["SWE"].mean()
    station_std_swe = tr.groupby("Station")["SWE"].std()

    station_stats_dict = {}
    for station in station_mean_swe.index:
        station_stats_dict[station] = (
            float(station_mean_swe[station]),
            float(station_std_swe[station]),
        )

    sample0 = next(iter(train_loader))
    cfg.input_size = sample0["dynamic forcing"].shape[-1]

    model = SWE_Net(cfg, station_stats=station_stats_dict).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.7
    )

    hist_mean_model = HistoricalMean(
        climo_lookup=climo.to_dict(),
        station_mean=station_mean.to_dict(),
    ).to(device)
    hist_mean_model.eval()

    best_val_nse = float("-inf")
    best_epoch = -1
    best_model_state = None
    conformal_alpha = float(getattr(cfg, "conformal_alpha", 0.1))

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
                    loss = masked_mse(outputs, y_target, mask)
                elif cfg.loss.upper() == "NSE":
                    loss = masked_nse(outputs, y_target, mask)
                else:
                    raise ValueError(f"Unknown loss: {cfg.loss}")
            elif isinstance(cfg.loss, (list, tuple)):
                weights_ = cfg.loss_weights
                terms = []
                for w, name in zip(weights_, cfg.loss):
                    name = name.upper()
                    if name == "MSE":
                        terms.append(w * masked_mse(outputs, y_target, mask))
                    elif name == "NSE":
                        terms.append(w * masked_nse(outputs, y_target, mask))
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
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                os.makedirs("results", exist_ok=True)
                torch.save(best_model_state, "results/best_model.pt")
                print(f"*** New best model saved (Epoch {best_epoch}, Val NSE = {best_val_nse:.4f}) ***")

    print("\nCalibrating split-conformal intervals on VALIDATION set...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif os.path.exists("results/best_model.pt"):
        model.load_state_dict(torch.load("results/best_model.pt", map_location=device))

    calibration_df = collect_predictions(
        model=model,
        loader=val_loader,
        hist_mean_model=hist_mean_model,
        swe_normalizers=swe_normalizers,
        backtrans_cache=backtrans_cache,
        station_index=station_index,
        weights=weights,
        swe_lookup=swe_lookup,
        cfg=cfg,
    )
    conformal_global_q_hat = conformal_quantile(calibration_df, conformal_alpha)
    conformal_q_hat_by_station = conformal_quantiles_by_station(calibration_df, conformal_alpha)
    station_q_hats = np.array(list(conformal_q_hat_by_station.values()), dtype=float)
    print(
        f"Validation station-specific calibration residual quantiles "
        f"(alpha={conformal_alpha:.3f}, target coverage={1.0 - conformal_alpha:.3f})"
    )
    print(
        f"Stations calibrated: {len(conformal_q_hat_by_station)} | "
        f"median q_hat: {np.median(station_q_hats):.4f} | "
        f"range: {np.min(station_q_hats):.4f} to {np.max(station_q_hats):.4f} | "
        f"global fallback q_hat: {conformal_global_q_hat:.4f}"
    )

    test_results = run_test(
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
        cfg=cfg,
        conformal_q_hat_by_station=conformal_q_hat_by_station,
        conformal_global_q_hat=conformal_global_q_hat,
        conformal_alpha=conformal_alpha,
    )

    total_time = time.time() - total_start_time
    print(f"\nTotal Time: {total_time / 60:.2f} minutes")

    return model, test_results, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = SimpleNamespace(**cfg_dict)
    model, test_results, total_time = train_model(cfg)

    import csv
    from datetime import datetime
    from filelock import FileLock

    summary_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(summary_dir, exist_ok=True)

    summary_path = os.path.join(summary_dir, "sweep_summary.csv")
    lock = FileLock(summary_path + ".lock")

    row = {
        "timestamp": datetime.now().isoformat(),
        "run_name": getattr(cfg, "run_name", None),
        "M": getattr(cfg, "M", None),
        "test_nse": test_results["test_nse"],
        "test_rmse": test_results["test_rmse"],
        "total_time_seconds": total_time,
    }

    with lock:
        file_exists = os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
