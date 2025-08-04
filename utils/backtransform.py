from SpatialTransform import SpatialTransformer
import numpy as np
def back_transform_predictions(y_pred_decor: np.ndarray, transformer: SpatialTransformer, spatial_obj: dict) -> np.ndarray:
    y_pred_spatial = transformer.back_transform_to_spatial(y_pred_decor, spatial_obj)
    return np.array(y_pred_spatial)  