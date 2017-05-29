"""
 Advanced Econometrics 2 Problem set 2 Problem 1 OLS. Author: Mate Kormos
 data files was prepared with matlab from birthdata.txt.
"""

# Import dependencies
import numpy as np
import scipy.io as sio
import statsmodels.api as sm

# Get the data from matlab .m files
yraw = sio.loadmat('y.mat')
draw = sio.loadmat('d.mat')
Xraw = sio.loadmat('X.mat')
y = yraw['y']
d = draw['d']
X = Xraw['x']
print('Unconditional mean of smoking, full sample: ', d.mean())


# Restict the sample to facilitate analysis
subsamplesize = 60000
# seed random
np.random.seed([0])
# random permutation
permindex = np.random.permutation(len(y))
y_sub = y[permindex[0:subsamplesize]]
d_sub = d[permindex[0:subsamplesize]]
X_sub = X[permindex[0:subsamplesize], :]

# OLS
# merge d and X
Xd_sub = np.concatenate((d_sub, X_sub), axis=1)
# add constant as first column (d gets to the second column)
const = np.ones((subsamplesize, 1))
Xd_sub_cons = np.concatenate((const, Xd_sub), axis=1)
# estimate beta
XTX = (Xd_sub_cons.T).dot(Xd_sub_cons)
XTX_inv = np.linalg.inv(XTX)
b1 = XTX_inv.dot(Xd_sub_cons.T)
bhat = b1.dot(y_sub)
print('Estimated coeff vector=\n', bhat)
# residuals
uhat = y_sub - Xd_sub_cons.dot(bhat)
# estimate covmatrix not robust
s2hat = (uhat.T).dot(uhat) / (subsamplesize - 14)
covhat = s2hat * XTX_inv
print('Estimated cov matrix=\n', covhat)
# check with built-in function
results = sm.OLS(y_sub, Xd_sub_cons).fit()
print(results.summary())
