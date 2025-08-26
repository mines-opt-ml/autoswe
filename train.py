import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_squared_error

from dataloader import SWEDataLoader, SWEStationDataset
from modelzoo.NeuralNet import SWE_Net
from utils.backtransform import back_transform_scalar_with_weights


def train_model(cfg: SimpleNamespace) -> SWE_Net:
    device = torch.device(cfg.device)

    dataset = SWEStationDataset(cfg)
    orig_swe = dataset.dynamic_forcing_and_swe.reset_index()
    swe_lookup = orig_swe.set_index(["Station", "Date"])["SWE"]

    # ToDo: implement code below properly.
    if first_run:
        dataloader = SWEDataLoader(cfg)
        train_loader, val_loader, _, bt_info = dataloader.prepare()
        pickle.dump((train_loader, val_loader, _, bt_info), open("backtransform_info.pkl", "wb"))
    else:
        train_loader, val_loader, _, bt_info = pickle.load(open("backtransform_info.pkl", "rb"))

        station_index = bt_info["station_index"]
        weights = bt_info["weights"]
    backtrans_cache = bt_info["backtrans_cache"]

    model = SWE_Net(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    all_predictions = []

    for epoch in range(cfg.n_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg.n_epochs}")
        print("-" * 50)

        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            X = batch["dynamic forcing"].to(device)
            y = batch["swe"].to(device)

            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                X = batch["dynamic forcing"].to(device)
                mask = batch["mask"]
                stations = batch["station"]
                dates = batch["dates"]

                preds = model(X).cpu().numpy()

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
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val RMSE: {rmse:.4f}")
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

        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        pred_file = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(pred_file, index=False)

        print(f"Saved predictions to {pred_file}")

    return model


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    model = train_model(cfg)
