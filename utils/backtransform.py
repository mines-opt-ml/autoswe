from utils.SpatialTransform import SpatialTransformer
import numpy as np

# Original - from Heaton, et. al (2024)
def back_transform_predictions(y_pred_decor: np.ndarray, transformer: SpatialTransformer, spatial_obj: dict) -> np.ndarray:
    y_pred_spatial = transformer.back_transform_to_spatial(y_pred_decor, spatial_obj)
    return np.array(y_pred_spatial)  

# New - back-transform using precomputed weights + cached offsets 
def back_transform_scalar_with_weights(pred_prime: float, station_idx: int, date_str: str,
                                       weights, backtrans_cache: dict[str, np.ndarray]) -> float:
    return float(pred_prime * np.sqrt(weights.w[station_idx]) + backtrans_cache[date_str][station_idx])