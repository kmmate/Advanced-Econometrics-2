"""
Robust standard error estimation for linear models, clustered standard errors.
"""


def robustse_bootstrap(xdata, ydata, groupid, bootsamplenumber):
    """
    Block bootstrap estimator of covariance matrix with clustered standard errors.

    Estimates the covariance matrix of a model, where the model and the data is such that ydata = xdata.dot(beta) + u,
    with clustered structure.

    PARAMETERS
    ---------------
    :param xdata: sample data, NxK array of regressors. N: total number of observations (summed across all groups),
                  K: number of regressors (with potential constant included)
    :param ydata: sample data, N-array of independent variable
    :param groupid: N-array of group id. For each observation groupid contains to which group it belongs to
    :param bootsamplenumber: number of bootstrap resamples

    RETURNS
    --------------
    :return: clustered standard error covariance matrix
    """

    # import dependencies
    import numpy as np

    # errors
    if len(ydata) != len(xdata):
        raise Exception('The length of xdata and ydata must be the same.')
    if len(ydata) != len(groupid):
        raise Exception('The length of groupid and ydata must be the same.')

    # size, dimensions
    n = len(ydata)
    try:
        d = np.size(xdata, 1)
    except:
        d = 1

    # group informations
    # get the unique group ids
    groups = np.unique(groupid)
    # number of groups
    G = len(groups)
    # associate each gruop id with an integer if groupid is an array of strings
    if isinstance(groupid[0], str):
        # dictionary of correspondence: groupid to integer
        groupid_dict = dict()
        for g in range(G):
            groupid_dict[groups[g]] = g
        # new group id array with groupids indicated with integers
        groupid_integer = np.array([groupid_dict[i] for i in range(n)])
    # if the gropid is given as integers, use that
    else:
        groupid_integer = groupid

    # pre-allocate bhatmatrix, which contains the estimated coeff vector for each bootstrap sample
    bhatmatrix = np.empty((d, bootsamplenumber))
    # seed
    np.random.seed([0])
    # resampling
    for bootsample in range(bootsamplenumber):
        # draw a random sample of G clusters
        # random sample of G groups
        groupindex = np.random.randint(low=0, high=G, size=G)
        # loop through the sampled group indices and append the elments with the given index to the new data matrix
        counter = 0
        for sample_idx in groupindex:
            # masking
            mask = [i == sample_idx for i in groupid_integer]
            # the bootstrap sample
            # create when it's the first element in groupindex
            if counter == 0:
                y_boot = ydata[mask]
                x_boot = xdata[mask]
            # if not, append
            else:
                y_boot = np.concatenate((y_boot, ydata[mask]), axis=0)
                x_boot = np.concatenate((x_boot, xdata[mask]), axis=0)
            counter += 1
        # estimate beta
        xtxinv = np.linalg.inv((x_boot.T).dot(x_boot))
        bhat_p1 = xtxinv.dot(x_boot.T)
        bhat_boot = bhat_p1.dot(y_boot)
        # load to matrix
        bhatmatrix[:, bootsample] = bhat_boot

     # compute the estimated covariance matrix based on sample covariance
    varhat = np.cov(bhatmatrix)
    # return
    return  varhat
