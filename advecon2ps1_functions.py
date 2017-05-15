# Advanced Econometrics 2 Problem set 1. Author: Mate Kormos.
def spectral_density(omega, data, usebartlett=False, q=None):
    """
    Estimates the spectral density of the sample data at the given frequency
    :param omega: frequency: S_hat(omega) is returned
    :param data: sample data
    :param usebartlett: use Bartlett kernel for downweighting distant covariances (recommended)
    :param q: cutoff to the Barlett kernel
    :return: S_hat(omega)
    """
    # Import dependencies
    import numpy as np

    # Raise error
    if usebartlett and q is None:
        raise Exception('If usebarlett=True, q has to be given')

    # Get size
    n = len(data)
    # Define covariance estimator

    def autocovest(y, h):
        """
        Estimates the autocovariance of order h of sample data
        :param y: sample data
        :param h: order of autocovariance
        :return:
        """
        n = len(y)
        ymean = y.mean()
        gammahat = 1 / n * sum([(y[t] - ymean) * (y[t-h] - ymean) for t in range(h, n)])
        return gammahat

    # Define Barlett kernel
    def bartlett(u, cutoff):
        """
        Barlett kernel
        :param u: input to kernel
        :param cutoff: cut-off point after which the kernel value is zero
        :return:
        """
        if u <= cutoff:
            balett = 1 - u / (cutoff + 1)
        else:
            balett = 0
        return balett

    # Estimate the spectral density
    if usebartlett:
        sdhat = 1 / (2 * np.pi) * (autocovest(data, 0) + 2 *
                                   sum([autocovest(data, h=h) * bartlett(h, cutoff=q) * np.cos(omega * h)
                                        for h in range(1, q + 2)]))
    else:
        sdhat = 1 / (2 * np.pi) * (autocovest(data, 0) + 2 *
                                   sum([autocovest(data, h=h) * np.cos(omega * h) for h in range(1, n + 1)]))
    return sdhat


def customdiff(y, p):
    """
    Produces a differenced version of a series y, described by the lag polynomial (1-L^p)y
    :param y: time series
    :param p: the exponent of L: (1-L^p)
    :return: the lagged version of the series
    """
    # Import dependencies
    import numpy as np
    # Get size
    n = len(y)
    # Empty array
    y_diff = np.empty((n, ))
    # First values are nans
    y_diff[0:p] = np.nan
    for t in range(p, n):
        y_diff[t] = y[t] - y[t-p]
    return y_diff
