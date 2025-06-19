# %%
# import rpy2.robjects as robjects
# from rpy2.robjects import r, FloatVector
# import rpy2.robjects.packages as rpackages
# from rpy2.robjects import numpy2ri
# import pandas as pd
# import numpy as np


# def mkNNindx(locs, M):

#     numpy2ri.activate()

#     locs = robjects.r["matrix"](locs, nrow=len(locs))

#     cran_mirror_url = "https://cran.r-project.org/"

#     # Set the CRAN mirror option in R
#     robjects.r.options(repos=cran_mirror_url)

#     utils = rpackages.importr("utils")
#     utils.install_packages("spNNGP")

#     knnindx = """

#         library(spNNGP)


#         mkNNIndx <- function(coords, m, n.omp.threads=1){

#             get.n.indx <- function(i, m){
#                 i <- i-1
#                 if(i == 0){
#                 return(NA)
#                 }else if(i < m){
#                     n.indx.i <- i/2*(i-1)
#                     m.i <- i
#                     return((n.indx.i+1):((n.indx.i+1)+i-1))
#                 }else{
#                     n.indx.i <- m/2*(m-1)+(i-m)*m
#                     m.i <- m
#                     return((n.indx.i+1):((n.indx.i+1)+m-1))
#                 }
#         }

#         n <- nrow(coords)
#         nIndx <- (1+m)/2*m+(n-m-1)*m
#         nnIndx <- rep(0, nIndx)
#         nnDist <- rep(0, nIndx)
#         nnIndxLU <- matrix(0, n, 2)

#         n <- as.integer(n)
#         m <- as.integer(m)
#         coords <- as.double(coords)
#         nnIndx <- as.integer(nnIndx)
#         nnDist <- as.double(nnDist)
#         nnIndxLU <- as.integer(nnIndxLU)
#         n.omp.threads <- as.integer(n.omp.threads)

#         out <- .Call("mkNNIndx", n, m, coords, nnIndx, nnDist, nnIndxLU, n.omp.threads)

#         n.indx <- as.integer(nnIndx)

#         n.indx.list <- vector("list", n)
#         n.indx.list[1] <- NA
#         for(i in 2:n){
#             n.indx.list[[i]] <- n.indx[get.n.indx(i, m)]+1
#         }
#         n.indx.list
#     }

#            """

#     robjects.r(knnindx)

#     # Call the R function from Python
#     result = robjects.r["mkNNIndx"](locs, M)

#     # subtract 1 from each element in each index of the array to get the 0-based index
#     result = [np.array([i - 1 for i in x]) for x in result]
#     # but keep the NA in the 0th index
#     result[0] = np.array([np.nan])
#     # then make them all the elements of the array integers expect for the NA
#     result = [x.astype(int) for x in result]

#     return result


import numpy as np
from joblib import Parallel, delayed
import multiprocessing


def mkNNindx(locs, K):
    def calculate_neighbors(ind):
        if ind == 0:
            return np.nan
        else:
            dists = np.sqrt(np.sum((locs[ind] - locs[:ind]) ** 2, axis=1))
            return np.argsort(dists)[: min(ind, K)]

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(calculate_neighbors)(i) for i in range(len(locs))
    )
    return results


if __name__ == "__main__":
    mkNNindx()
