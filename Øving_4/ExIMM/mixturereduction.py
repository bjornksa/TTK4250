from typing import Tuple

import numpy as np


def gaussian_mixture_moments(
    w: np.ndarray,  # the mixture weights shape=(N,)
    mean: np.ndarray,  # the mixture means shape(N, n)
    cov: np.ndarray,  # the mixture covariances shape (N, n, n)
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the mean and covariance of of the mixture shapes ((n,), (n, n))
    """Calculate the first two moments of a Gaussian mixture"""

    # mean
    mean_bar = np.average(mean, weights=w, axis=0)

    # covariance
    # # internal covariance
    cov_int = np.average(cov, weights=w, axis=0)

    # # spread of means
    # Optional calc: mean_diff =
    mean_diff = np.diagonal((mean - mean_bar)@(mean - mean_bar).T)
    cov_ext = np.average(mean_diff, weights=w, axis=0)

    # # total covariance
    cov_bar = cov_int + cov_ext

    return mean_bar, cov_bar
