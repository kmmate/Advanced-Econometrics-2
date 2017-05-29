"""
Matcher and matching estimator functions
"""
def matching_matcher(matchtothis, matchfromthis, covhat, distancefunction='L2cov', matchtype='knearest',
            k=None, threshold=None, getmatchstats=False):
    """
    Matcher function

    Parameters
    ----------
    :param matchtothis: a vector or scalar to which the function look for matches
    :param matchfromthis: a vector or matrix in which the matches are searched
    :param covhat: estimated covariance matrix of the data
    :param distancefunction: string, name of distance function in distance_functions.py
    :param matchtype: 'knearest' or 'threshold', once the distance are found, the mode of matching is determined by
                       this
    :param k: number of neighbours in knearest matching
    :param threshold: threshold value in thereshold matching
    :param getmatchstats: if True, the function also returns a dictionary with descriptive statistics of the distance
    between matchtothis and the matched elements
    Returns
    ---------
    :return: matched indices: indices of the elements in matchfromthis that are matched to mathcedtothis or
    a tuple: these indices and matchedstats
    """
    # imports
    import numpy as np
    import distance_functions

    # errors
    if matchtype == 'threshold' and threshold is None:
        raise Exception('For matchtype=threshold, threshold must be given')
    if matchtype == 'knearest' and k is None:
        raise Exception('For matchtype=knearest, k must be given')

    # get the distance function
    distfunction = getattr(distance_functions, distancefunction)

    # compute the distances between mathtothis and each element in matchfromthis
    distances = np.array([distfunction(matchtothis, m, covhat) for m in matchfromthis])

    # match as specified
    if matchtype == 'threshold':
        # indices of elements in matchfromthis such that the distance of the element from mathtothis is smaller than
        #  or equal to threshold
        matchedindices = [i for i in range(len(matchfromthis)) if distances[i] <= threshold]
    elif matchtype == 'knearest':
        # indices of the k nearest element to matchtothis in matchfromthis
        matchedindices = np.argsort(distances)[0:k]

    # return the indices
    if not getmatchstats:
        return matchedindices
    else:
        # compute stats and put in distionary
        matchstats = {}
        matchstats['mean_dist'] = distances[matchedindices].mean()
        matchstats['min_dist'] = distances[matchedindices].min()
        matchstats['max_dist'] = distances[matchedindices].max()
        return matchedindices, matchstats


def matching_matching_est(effect, ydata, ddata, xdata, matchtype='knearest', k=None, threshold=None,
                          distancefunction='L2cov', getSE=False, getmatchstats=False, matchstatsmode='mean_dist'):
    """
    Matching estimator of ATE and ATT.

    Estimates the average treatment effect (ATE) and the average treatment effect for the treated (ATT) of a
    binary treatment.

    Parameters
    -----------
    :param effect: 'ATE' or 'ATT': average treatment effect or average treatment effect for the treated, respectively
    :param ydata: data on the outcome of interest, n*d array
    :param ddata: data on the treatment, n*d array
    :param xdata: data on covariates to match
    :param matchtype: 'knearest' or 'threshold': k-nearest neighbour matching or hard thresholding
    :param k: number of neighbours if matchtype='knerest'
    :param threshold: hard threshold to apply when matchtype='threshold'
    :param distancefunction: name of the distance function in distance_functions.py, string
    :param getSE: if True, standard errors are also returned
    :param getmatchstats: if True also returns a dictionary of the specified property, specified by matchstatsmode,
                          of the distances between match-to and matched elements
    :param matchstatsmode: 'mean_dist' or 'min_dist' or 'max_dist' if True descriptive stats are also returned on the
    mean distance or minimum distance or the maximum distance (respectively) between the match-to and matched elements

    Returns
    ------------
    :return: estimatorhat or (estimatorhat, se) or (estimatorhat, se, matchstats)
    """

    # imports
    import numpy as np
    # size
    n = len(ydata)
    # treated and control group indices, sizes
    t_idx = [i for i in range(n) if ddata[i] == 1]
    c_idx = [i for i in range(n) if ddata[i] == 0]
    n_t = len(t_idx)
    n_c = len(c_idx)
    # values
    y_t = ydata[t_idx]
    y_c = ydata[c_idx]
    x_t = xdata[t_idx]
    x_c = xdata[c_idx]

    # sample covariance matrix
    if distancefunction == 'L2cov':
        covhat = np.cov(xdata, rowvar=False)

    # ATT
    if effect == 'ATT':
        # list of the differences
        delta_t_list = list()
        if getmatchstats:
            matchstat_t_list = list()
        # element in treated group
        for t in range(n_t):
            # get the matched indices and the matchstat (if required) for element t
            if getmatchstats:
                matches_t, matchstat_t = matching_matcher(matchtothis=x_t[t], matchfromthis=x_c, covhat=covhat,
                                                          distancefunction=distancefunction, matchtype=matchtype,
                                                          k=k, threshold=threshold, getmatchstats=True)
                # append to list the desired property of the distances
                matchstat_t_list.append(matchstat_t[matchstatsmode])
            else:
                matches_t = matching_matcher(matchtothis=x_t[t], matchfromthis=x_c, covhat=covhat,
                                                          distancefunction=distancefunction, matchtype=matchtype,
                                                          k=k, threshold=threshold, getmatchstats=False)
            ybar_t = y_c[matches_t].mean()
            delta_t = y_t[t] - ybar_t
            delta_t_list.append(delta_t)
        # produce descriptives on the list of desired property of distances
        if getmatchstats:
            matchstat = dict()
            matchstat['mean'] = np.array(matchstat_t_list).mean()
            matchstat['min'] = np.array(matchstat_t_list).min()
            matchstat['max'] = np.array(matchstat_t_list).max()
            matchstat['std'] = np.array(matchstat_t_list).std()
        # estimation
        atthat = np.array(delta_t_list).mean()
        # if SEs are required compute them
        if getSE:
            varhat_att = 1 / n_t ** 2 * sum([(delta - atthat) ** 2 for delta in delta_t_list])
            sehat_att = np.sqrt(varhat_att)[0]
        # possible returns
        if not getSE and not getmatchstats:
            return atthat
        if getSE and not  getmatchstats:
            return atthat, sehat_att
        if not getSE and getmatchstats:
            return atthat, matchstat
        if getSE and getmatchstats:
            return atthat, sehat_att, matchstat
    # ATE
    elif effect == 'ATE':
        # estimated y's
        yhat_t = np.empty((n, ))
        yhat_c = np.empty((n, ))
        # all elements
        for j in range(n):
            # if j is treated
            if ddata[j] == 1:
                # then the estimated treated counterfactual Y_j(1) for j is the true y_j
                yhat_t[j] = ydata[j]
                # and the estimated control counterfactual Y_j(0) for j is based on matching
                mathces = matching_matcher(matchtothis=xdata[j], matchfromthis=x_c, covhat=covhat,
                                           distancefunction=distancefunction, matchtype=matchtype, k=k,
                                           threshold=threshold)
                yhat_c[j] = y_c[mathces].mean()
            # if j is not treated
            else:
                # then the estimated treated counterfactual Y_j(1) for j is based on mathching
                mathces = matching_matcher(matchtothis=xdata[j], matchfromthis=x_t, covhat=covhat,
                                           distancefunction=distancefunction, matchtype=matchtype, k=k,
                                           threshold=threshold)
                yhat_t[j] = y_t[mathces].mean()
                # and the estimated control counterfactual Y_j(0) for j is the true y_j
                yhat_c[j] = ydata[j]
        # estimation
        atehat = (yhat_t - yhat_c).mean()
        # standard errors
        if getSE:
            varhat_ate = 1 / n ** 2 * sum([(yhat_t[j] - yhat_c[j] - atehat) ** 2 for j in range(n)])
            sehat_ate = np.sqrt(varhat_ate)
            return atehat, sehat_ate
        else:
            return atehat