"""
Distance function used with matching.py
"""
def L2cov(v1, v2, covhat):
    """
    L2 norm distance function, adjusted for variance in variables
    :param v1: distance of this vector...
    :param v2: ...from thiis vector
    :param covhat: estimated covariance
    :return: scalar, variance-adjusted L2 norm
    """
    # Import depenndencies
    import numpy as np
    # Compute the distance term-by-term
    # try if there are multiple variables
    try:
        covhat_inv = np.linalg.inv(covhat)
        vectordist = v1[:, None] - v2[:, None]
        dist1 = (vectordist.T).dot(covhat_inv)
        distance = dist1.dot(vectordist)
        return distance[0, 0]
    # if not, go with
    except:
        distance = np.abs(v1 - v2) ** 2 / covhat
        return distance
