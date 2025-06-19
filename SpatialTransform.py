import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from scipy.linalg import cholesky
import mknnIndx
import matern
from scipy.linalg import cholesky


def process_row(
    idx, ytrain, Xtrain, trainLocs, nnList, smoothness, range_param, nugget
):
    if idx == 0:
        y = ytrain.iloc[idx]
        w = 1
        X = Xtrain.iloc[idx] / np.sqrt(w)
    else:
        locs = (
            trainLocs[: idx + 1] if idx == 1 else trainLocs[np.append(idx, nnList[idx])]
        )
        D = cdist(locs, locs)

        covariance_matrix = matern.Matern(D, range_param, smoothness, phi=1.0)
        R = (1 - nugget) * covariance_matrix + (nugget * np.eye(D.shape[0]))

        R_inv = np.linalg.inv(R[1:, 1:])
        w = 1 - np.dot(R[0, 1:], np.dot(R_inv, R[1:, 0]))
        X = (
            Xtrain.iloc[idx].T - R[0, 1:].dot(R_inv.dot(Xtrain.iloc[nnList[idx]]))
        ) / np.sqrt(w)
        y = (
            ytrain.iloc[idx] - R[0, 1:].dot(R_inv.dot(ytrain.iloc[nnList[idx]]))
        ) / np.sqrt(w)

    return {"y": y, "X": X, "w": w}


def process_test_data(
    idx, testLocs, trainLocs, Xtest, Xtrain, ytrain, nugget, range_param, smoothness, M
):

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
    def __init__(self):
        pass

    def transform_to_ind(
        self,
        target,
        trainData,
        trainLocs,
        testData,
        testLocs,
        smoothness=0.5,
        range_param=1,
        nugget=0.01,
        M=30,
        ncores=1,
    ):

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
            delayed(process_row)(
                idx, ytrain, Xtrain, trainLocs, nnList, smoothness, range_param, nugget
            )
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

        trainData_y = pd.DataFrame(
            np.vstack([x["y"] for x in indData]), columns=[target]
        )
        trainData_X = pd.DataFrame(
            np.vstack([x["X"] for x in indData]), columns=trainData_columns
        )
        testData_X = pd.DataFrame(
            np.vstack([x["X"] for x in indTestData]), columns=testData_columns
        )

        trainData_combined = pd.concat([trainData_y, trainData_X], axis=1)

        outList = {
            "trainData": trainData_combined,
            "testData": testData_X,
            "range": range_param,
            "nugget": nugget,
            "M": M,
            "backTransformInfo": [
                {"w": x["w"], "backTrans": x["backTrans"]} for x in indTestData
            ],
        }

        return outList

    def back_transform_to_spatial(self, preds, transformObj):
        spatialPreds = (
            preds
            * np.array(list(map(lambda x: x["w"], transformObj["backTransformInfo"])))
        ) + np.array(
            list(map(lambda x: x["backTrans"], transformObj["backTransformInfo"]))
        )

        return spatialPreds


# Example usage:
if __name__ == "__main__":
    transformer = SpatialTransformer()
    # Call methods on the transformer object as needed
