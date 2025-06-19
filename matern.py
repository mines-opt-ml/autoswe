# %%
import numpy as np
from scipy.special import gamma, kv


def Matern(d, range_param=1, nu=0.5, phi=1.0):
    """
    Matern covariance function transcribed from Stein's book page 31

    Parameters:
        d (array-like): Distances.
        range (float): Range parameter (default: 1).
        alpha (float): Scale parameter (default: 1).
        smoothness (float): Smoothness parameter (default: 0.5).
        nu (float): Smoothness parameter (overrides smoothness if provided).
        phi (float): Variance parameter (default: 1.0).

    Returns:
        array-like: Matern covariance values.
    """
    alpha = 1.0 / range_param
    # Check for negative distances
    if np.any(d < 0):
        raise ValueError("Distance argument must be nonnegative")

    # Rescale distances
    d = d * alpha

    # Call some special cases for half fractions of nu
    if nu == 0.5:
        return phi * np.exp(-d)
    if nu == 1.5:
        return phi * (1 + d) * np.exp(-d)
    if nu == 2.5:
        return phi * (1 + d + d**2 / 3) * np.exp(-d)

    # Otherwise ...
    # Avoid sending exact zeroes to kv
    d[d == 0] = 1e-10

    # The hairy constant
    con = (2 ** (nu - 1)) * gamma(nu)
    con = 1 / con

    return phi * con * (d**nu) * kv(nu, d)


if __name__ == "__main__":
    Matern()
