# Gaussian Process Decorrelation for Spatiotemporal Deep Learning Snow-Water Equivalent Prediction
## Colin D. Fenster<sup>1</sup>, Adrienne M. Marshall<sup>2</sup>, Soutir Bandyopadhyay<sup>1</sup>, Daniel McKenzie<sup>1</sup>
### <sup>1</sup> Colorado School of Mines, Department of Applied Mathematics and Statistics, Golden, CO. <sup>2</sup> Colorado School of Mines, Department of Geology and Geological Engineering, Golden, CO. 

Predicting Snow-Water Equivalent (SWE) using a joint spatial statistics and time-series deep learning approach using meteorological forcings and orographic features.

## Repository Structure

```text
autoswe/
├── data/                        
├── tests/
│   ├── conftest.py
│   └── dataset_test.py
├── utils/
│   ├── SpatialTransform.py        
│   ├── backtransform.py           
│   ├── matern.py                  
│   ├── metrics.py                 
│   ├── mknnIndx.py                
│   ├── normalizer.py             
│   ├── preprocess.py              
│   └── snowyear_DOY_conversion.py
├── config.yaml                    
├── dataloader.py                  
├── dataset.py                     
├── train.py                       
├── map_nse_results.py             
├── LICENSE
└── README.md
```

## Overview

The current model pipeline is:

```text
Raw SWE + meteorological forcing and orographic feature CSVs
        ↓
preprocess.py
        ↓
SWEStationDataset
        ↓
Spatial decorrelation of SWE and selected dynamic forcings
        ↓
LSTM sequence model
        ↓
Back-transform predictions
        ↓
Station-level and global evaluation
```

The main model is an LSTM that predicts SWE sequences for each station and snow year. Spatial dependence is handled through a nearest-neighbor Gaussian-process-style decorrelation step. After prediction, the model output is transformed back to the original SWE scale for evaluation.

## Data Requirements

The `data/` directory should contain the SWE, meteorological, remote-sensing, and metadata CSV files referenced in `config.yaml`.

Expected inputs include files for:

- SWE observations
- maximum temperature
- minimum temperature
- precipitation
- observed temperature
- passive microwave brightness temperature features
- SNOTEL station metadata

Station names are standardized internally by replacing underscores with spaces and converting to lowercase.

## Configuration

Experiments are controlled through `config.yaml`. Important settings include:

```yaml
train_start_year: 2001
train_end_year: 2012
val_start_year: 2013
val_end_year: 2015
test_start_year: 2016
test_end_year: 2018

beginning_of_snow_year: "10-01"
end_of_snow_year: "06-30"

batch_size: 16
shuffle: true
n_epochs: 40
lr: 0.001
device: "cuda:0"

M: 30
range_param: 1.0
smoothness: 0.5
nugget: 0.01

lag_days: 1
loss: "MSE"
normalization: "zscore"
```

Paths to input CSV files should also be specified in `config.yaml`.

## Running Training

From the repository root:

```bash
python train.py --config config.yaml
```

The script will:

1. load and preprocess the input CSV files,
2. construct station-year sequence samples,
3. apply spatial decorrelation,
4. train the LSTM model,
5. evaluate on validation data after each epoch,
6. save the best model by validation NSE,
7. evaluate the best model on the test set,
8. write prediction and metric CSVs to `results/`.

## Outputs

Training and testing outputs are saved in `results/`.

Common output files include:

```text
results/best_model.pt
results/test_predictions.csv
results/test_station_metrics.csv
results/test_predictions_with_baseline.csv
results/test_station_metrics_baseline.csv
results/test_predictions_persistence.csv
results/test_station_metrics_persistence.csv
results/sweep_summary.csv
```

`test_predictions.csv` contains model predictions, actual SWE values, station names, and dates.

`test_station_metrics.csv` contains station-level NSE, RMSE, and prediction counts.

## Baselines

The training script currently evaluates the LSTM against two baselines:

1. **Climatology baseline**  
   A per-station, per-day-of-year historical mean SWE estimate built from the training period.

2. **Persistence baseline**  
   A simple baseline where predicted SWE at time `t` is equal to observed SWE at time `t - 1`.

These baselines are exported alongside model predictions for comparison.

## Model Description

The core neural network is `SWE_Net`, defined in `modelzoo/LSTM.py`. It uses:

- an LSTM sequence model,
- dropout regularization,
- a final linear layer,
- an output scaling parameter,
- optional station-level context features based on station mean and standard deviation of SWE.

Input features are constructed in `dataset.py` and include:

- elevation,
- slope,
- aspect,
- latitude,
- longitude,
- day of snow year,
- selected dynamic meteorological/remote-sensing features,
- optional lagged SWE terms.

## Spatial Decorrelation

Spatial decorrelation is implemented in `utils/SpatialTransform.py`.

The current approach computes nearest-neighbor GP-style weights using a Matérn covariance function. For each station, the response and selected dynamic features are transformed using nearby stations and a conditional variance term. This creates approximately spatially decorrelated training targets and inputs.

Predictions are mapped back to the original spatial scale using cached neighbor offsets in `utils/backtransform.py`.

## Testing

Run the test suite with:

```bash
pytest tests/
```

These tests are not necessary for model instantiation and are purely for developmental verifications.

## Citation Notes

This codebase is motivated by methods from Gaussian-process spatial statistics, nearest-neighbor Gaussian processes, and deep learning for hydrologic prediction. Key methodological references include:

[1] Matthew J. Heaton, Andrew Millane, and Jake S. Rhodes. A Scalable Spatial Decorrelation Preprocessing Approach for Machine and Deep Learning. Journal of Data Science, 2025. 

[2] Abhirup Datta, Sudipto Banerjee, Andrew O. Finley, and Alan E. Gelfand. Hierarchical Nearest-Neighbor Gaussian Process Models for Large Geostatistical Datasets. Journal of the American Statistical Association, 2016.

[3] F. Kratzert, M. Gauch, G. Nearing, and D. Klotz. NeuralHydrology — A Python library for Deep Learning research in hydrology. Journal of Open-Source Software, 2022.

## License

See `LICENSE` for licensing information.
