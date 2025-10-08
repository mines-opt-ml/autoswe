from dataclasses import dataclass
from typing import Any, Iterable, Optional
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
from utils import matern  

@dataclass
class PrecomputedWeights:
    nn_index: np.ndarray  
    A: np.ndarray         
    w: np.ndarray         

def precompute_weights(
    trainLocs: np.ndarray,
    M: int,
    range_param: float,
    smoothness: float,
    nugget: float
) -> PrecomputedWeights:
    """
    Precompute per-station neighbor weights (a_i) and scalars (w_i) once.
    Assumes trainLocs is ordered in the same station order that will be used
    when constructing (per-date) station vectors for the transform.
    """
    trainLocs = np.asarray(trainLocs, dtype=np.float32)
    N = int(trainLocs.shape[0])

    M_eff = int(min(max(1, M), max(1, N - 1)))

    tree = cKDTree(trainLocs)
    _, inds = tree.query(trainLocs, k=M_eff + 1)
    nn_index = inds[:, 1:].astype(np.int64)  

    A = np.empty((N, M_eff), dtype=np.float32)
    w = np.empty((N,), dtype=np.float32)

    I = np.eye(M_eff + 1, dtype=np.float32)

    for i in range(N):
        nbrs = nn_index[i]                      
        idxs = np.concatenate(([i], nbrs))       
        locs = trainLocs[idxs]                  

        D = cdist(locs, locs)
        K = matern.Matern(D, range_param, smoothness, phi=1.0).astype(np.float32)
        R = (1.0 - nugget) * K + nugget * I

        chol = cholesky(R[1:, 1:], lower=False, overwrite_a=False, check_finite=True)
        t = np.linalg.solve(chol.T, R[0, 1:])
        a = np.linalg.solve(chol, t).astype(np.float32)  

        wi = np.float32(1.0 - a @ R[1:, 0])
        wi = np.float32(max(wi, 1e-8))

        A[i, :] = a
        w[i] = wi

    return PrecomputedWeights(nn_index=nn_index, A=A, w=w)

def fast_transform_with_weights(
    trainData: pd.DataFrame,
    target: str,
    weights: PrecomputedWeights,
    cols_to_transform: Optional[Iterable[str]] = None,
    static_cols: Optional[Iterable[str]] = None,
    station_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply spatial transform with precomputed weights to the target AND the
    selected input columns (decorrelate inputs too).
    Returns:
        DataFrame with the same columns as trainData,
        where 'target' and selected input columns are decorrelated.
    """
    Nw = weights.nn_index.shape[0]

    if static_cols is None:
        static_cols = [
            "Elevation", "Slope", "Aspect", "Latitude", "Longitude",
            "Elevation_x", "Slope_tif1_x", "Aspect_tif_x", "Latitude_x", "Longitude_x",
            "Elevation_y", "Slope_tif1_y", "Aspect_tif_y", "Latitude_y", "Longitude_y",
        ]

    if cols_to_transform is None:
        numeric_cols = trainData.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_transform = [c for c in numeric_cols if c != target and c not in static_cols]

    out = trainData.copy()

    y = out[target].to_numpy(dtype=np.float32)         
    N = y.shape[0]
    M = weights.nn_index.shape[1]

    y_out = np.empty_like(y, dtype=np.float32)
    for i in range(N):
        a = weights.A[i]                 
        nbrs = weights.nn_index[i]       
        denom = np.sqrt(weights.w[i])    
        y_out[i] = (y[i] - np.dot(a, y[nbrs])) / denom
    out[target] = y_out

    if cols_to_transform:
        X = out[cols_to_transform].to_numpy(dtype=np.float32)  
        X_out = np.empty_like(X, dtype=np.float32)

        for i in range(N):
            a = weights.A[i]               
            nbrs = weights.nn_index[i]     
            denom = np.sqrt(weights.w[i])  
            X_out[i, :] = (X[i, :] - a @ X[nbrs, :]) / denom

        out.loc[:, cols_to_transform] = X_out

    return out

def process_row(
    idx: int,
    ytrain: pd.Series,
    Xtrain: pd.DataFrame,
    trainLocs: np.ndarray,
    nnList: list[list[int]],
    smoothness: float,
    range_param: float,
    nugget: float,
) -> dict[str, Any]:
    """
    Reference (slower) single-row transform that recomputes local Matern per row.
    Useful for testing; the fast path should be used for production.
    """
    nbrs = nnList[idx]
    idxs = np.concatenate(([idx], np.asarray(nbrs, dtype=np.int64)))
    locs = trainLocs[idxs]
    D = cdist(locs, locs)

    covariance_matrix = matern.Matern(D, range_param, smoothness, phi=1.0).astype(np.float32)
    R = (1.0 - nugget) * covariance_matrix + (nugget * np.eye(D.shape[0], dtype=np.float32))

    if R.shape[0] == 1:
        y = np.float32(ytrain.iloc[idx])
        X = Xtrain.iloc[idx].to_numpy(dtype=np.float32)
        w = np.float32(1.0)
        return {"y": y, "X": X, "w": w}

    chol = cholesky(R[1:, 1:], lower=False, overwrite_a=False, check_finite=True)
    t = np.linalg.solve(chol.T, R[0, 1:])
    a = np.linalg.solve(chol, t).astype(np.float32)  

    wi = np.float32(1.0 - a @ R[1:, 0])
    wi = np.float32(max(wi, 1e-8))  

    denom = np.sqrt(wi).astype(np.float32)
    Xnbr = Xtrain.iloc[nbrs].to_numpy(dtype=np.float32)  
    Xin = Xtrain.iloc[idx].to_numpy(dtype=np.float32)    
    Yin = np.float32(ytrain.iloc[idx])
    Ynbr = ytrain.iloc[nbrs].to_numpy(dtype=np.float32)  

    X_out = (Xin - a @ Xnbr) / denom
    y_out = (Yin - a @ Ynbr) / denom

    return {"y": y_out, "X": X_out, "w": wi}


# End of new code; Heaton. et. al (2024) code below


def process_test_data(
    idx: int,
    testLocs: np.ndarray,
    trainLocs: np.ndarray,
    Xtest: pd.DataFrame,
    Xtrain: pd.DataFrame,
    ytrain: pd.Series,
    nugget: float,
    range_param: float,
    smoothness: float,
    M: int,
) -> dict[str, Any]:
    # Distance between test location and training locations
    D = cdist(testLocs[idx].reshape(1, -1), trainLocs)
    # Find the M nearest neighbors
    the_neighbors = np.argsort(D)[0][:M]

    # Distance Matrix
    R = cdist(
        np.vstack((testLocs[idx], trainLocs[the_neighbors])),
        np.vstack((testLocs[idx], trainLocs[the_neighbors])),
    )

    # Create Matern kernel with specified parameters
    covariance_matrix = matern.Matern(R, range_param, smoothness, phi=1.0)

    # Covariance matrix with nugget effect
    R = nugget * np.eye(M + 1) + (1 - nugget) * covariance_matrix

    # Cholesky decomposition and inversion
    chol = cholesky(R[1:, 1:], lower=False)
    chol_inv = np.linalg.inv(chol.T)
    R_inv = chol_inv.T @ chol_inv

    # Calculate the weights
    R12 = np.dot(R[0, 1:], R_inv)
    w = 1 - R12 @ (R[1:, 0])

    # Transform the test data
    X = (Xtest.iloc[idx].T - (R12 @ Xtrain.iloc[the_neighbors])) / np.sqrt(w)

    # Return the transformed data
    return {"backTrans": R12.dot(ytrain.iloc[the_neighbors]), "X": X, "w": w}


class SpatialTransformer:
    def __init__(self) -> None:
        pass

    def transform_to_ind(
        self,
        target: str,
        trainData: pd.DataFrame,
        trainLocs: np.ndarray,
        testData: pd.DataFrame,
        testLocs: np.ndarray,
        smoothness: float = 0.5,
        range_param: float = 1.0,
        nugget: float = 0.01,
        M: int = 30,
        ncores: int = 1,
    ) -> dict[str, Any]:
        nnList = mknnIndx.mkNNindx(trainLocs, M)

        ytrain = trainData[target]
        Xtrain = trainData.drop(columns=[target])
        Xtrain.insert(0, "Intercept", 1)

        Xtest = testData.drop(columns=[target])
        Xtest.insert(0, "Intercept", 1)

        trainData_columns = Xtrain.columns
        testData_columns = Xtest.columns

        n_samples = len(Xtrain)

        indData = Parallel(n_jobs=ncores)(
            delayed(process_row)(idx, ytrain, Xtrain, trainLocs, nnList, smoothness, range_param, nugget)
            for idx in range(n_samples)
        )

        indTestData = Parallel(n_jobs=ncores)(
            delayed(process_test_data)(
                idx,
                testLocs,
                trainLocs,
                Xtest,
                Xtrain,
                ytrain,
                nugget,
                range_param,
                smoothness,
                M,
            )
            for idx in range(len(Xtest))
        )

        trainData_y = pd.DataFrame(np.vstack([x["y"] for x in indData]), columns=[target])
        trainData_X = pd.DataFrame(np.vstack([x["X"] for x in indData]), columns=trainData_columns)
        testData_X = pd.DataFrame(np.vstack([x["X"] for x in indTestData]), columns=testData_columns)

        trainData_combined = pd.concat([trainData_y, trainData_X], axis=1)

        outList = {
            "trainData": trainData_combined,
            "testData": testData_X,
            "range": range_param,
            "nugget": nugget,
            "M": M,
            "backTransformInfo": [{"w": x["w"], "backTrans": x["backTrans"]} for x in indTestData],
        }

        return outList

    def back_transform_to_spatial(self, preds: np.ndarray, transformObj: dict[str, Any]) -> np.ndarray:
        spatialPreds = (preds * np.array(list(map(lambda x: x["w"], transformObj["backTransformInfo"])))) + np.array(
            list(map(lambda x: x["backTrans"], transformObj["backTransformInfo"]))
        )

        return spatialPreds


# Example usage:
if __name__ == "__main__":
    transformer = SpatialTransformer()
    # Call methods on the transformer object as needed
