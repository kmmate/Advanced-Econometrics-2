"""
 Advanced Econometrics 2 Problem set 2 Problem 1 logit and others. Author: Mate Kormos
 data files was prepared with matlab from birthdata.txt.
"""

# Import dependencies
import numpy as np
import scipy.io as sio
import statsmodels.discrete.discrete_model as smdm
import matching as mtch

# Get the data from matlab .m files
yraw = sio.loadmat('y.mat')
draw = sio.loadmat('d.mat')
Xraw = sio.loadmat('X.mat')
y = yraw['y']
d = draw['d']
X = Xraw['x']

# Restict the sample to facilitate analysis
subsamplesize = 30000
# seed random
np.random.seed([0])
# random permutation
permindex = np.random.permutation(len(y))
y_sub = y[permindex[0:subsamplesize]]
d_sub = d[permindex[0:subsamplesize]]
X_sub = X[permindex[0:subsamplesize], :]

# Estimate propensity scores
# append constant
const = np.ones((subsamplesize, 1))
X_sub_cons = np.concatenate((const, X_sub), axis=1)
# predict prop score
prop_score = smdm.Logit(d_sub, X_sub_cons).fit().predict()

# Inverse probability weighting
# ATT
numerator = np.array([prop_score[i] * (y_sub[i] * d_sub[i] / prop_score[i] - (1 - prop_score[i]) * y_sub[i] /
                                       (1 - prop_score[i])) for i in range(subsamplesize)]).sum()
denominator = prop_score.sum()
atthat_invprob = numerator / denominator
print('Inverse probability weighted estimation of ATT: ', atthat_invprob)

# Matching, ATT
# 1 nearest neighbour, k = 1
# effect = 'ATT'
# k = 1
# print('\nATT k-nearest neighbour mathcing started with k=', k)
# atthat_k1, sehat_att_k1, matchstat_att_k1 = mtch.matching_matching_est(effect=effect, ydata=y_sub, ddata=d_sub,
#                                                                        xdata=prop_score, matchtype='knearest', k=k,
#                                                                        distancefunction='L2cov', getSE=True,
#                                                                        getmatchstats=True, matchstatsmode='mean_dist')
# print('ATT estimation, k-nearest neighbour (k= %d): %.4f' % (k, atthat_k1))
# print('ATT SE estimation, k-nearest neighbour (k= %d): %.4f' % (k, sehat_att_k1))
# print('ATT matching statistics (k=', k ,')\n:', matchstat_att_k1)
# # 5 nearest neighbour, k = 1
# k = 5
# print('\nATT k-nearest neighbour mathcing started with k=', k)
# atthat_k2, sehat_att_k2, matchstat_att_k2 = mtch.matching_matching_est(effect=effect, ydata=y_sub, ddata=d_sub,
#                                                                        xdata=prop_score, matchtype='knearest', k=k,
#                                                                        distancefunction='L2cov',  getSE=True,
#                                                                        getmatchstats=True, matchstatsmode='mean_dist')
# print('ATT estimation, k-nearest neighbour (k= %d): %.4f' % (k, atthat_k2))
# print('ATT SE estimation, k-nearest neighbour (k= %d): %.4f' % (k, sehat_att_k2))
# print('ATT matching statistics (k=', k ,')\n:', matchstat_att_k2)
#
# # Mathcing, ATE
# # 1 nearest neighbour, k = 1
# effect = 'ATE'
# k = 1
# print('\n#################\n ATE \n ##################')
# print('\nATE k-nearest neighbour mathcing started with k=', k)
# atehat_k1, sehat_ate_k1 = mtch.matching_matching_est(effect=effect, ydata=y_sub, ddata=d_sub,
#                                                      xdata=prop_score, matchtype='knearest', k=k,
#                                                      distancefunction='L2cov', getSE=True,
#                                                      getmatchstats=False, matchstatsmode='mean_dist')
# print('ATE estimation, k-nearest neighbour (k= %d): %.4f' %(k, atehat_k1))
# print('ATE SE estimation, k-nearest neighbour (k= %d): %.4f' %(k, sehat_ate_k1))
# # 5 nearest neighbour, k = 1
# k = 5
# print('\nATE k-nearest neighbour mathcing started with k=', k)
# atehat_k2, sehat_ate_k2 = mtch.matching_matching_est(effect=effect, ydata=y_sub, ddata=d_sub,
#                                                      xdata=prop_score, matchtype='knearest', k=k,
#                                                      distancefunction='L2cov',  getSE=True,
#                                                      getmatchstats=False, matchstatsmode='mean_dist')
# print('ATE estimation, k-nearest neighbour (k= %d): %.4f' %(k, atehat_k2))
# print('ATE SE estimation, k-nearest neighbour (k= %d): %.4f' %(k, sehat_ate_k2))

################################ WINSORISED #######################################
print('\n\n########################################################################################################\
        \n WINSORISED \n\
    ###############################################################################################################\n')
# winsorise data
# limits
lowlimit = 0.05
upperlimit = 0.95
# boolean mask
mask = np.all([(lowlimit <= prop_score), (prop_score <= upperlimit)], axis=0)
# apply mask
y_sub_win = y_sub[mask]
d_sub_win = d_sub[mask]
X_sub_win = X_sub[mask]
prop_score_win = prop_score[mask]

# print winsorised sample size
print('Winsorised sample size:', len(y_sub_win))

# Inverse probability weighting
numerator = np.array([prop_score_win[i] * (y_sub_win[i] * d_sub_win[i] / prop_score_win[i] -
                                           (1 - prop_score_win[i]) * y_sub_win[i] /  (1 - prop_score_win[i]))
                      for i in range(len(y_sub_win))]).sum()
denominator = prop_score_win.sum()
atthat_invprob = numerator / denominator
print('Inverse probability weighted estimation of ATT: ', atthat_invprob)


# Matching, ATT
# 1 nearest neighbour, k = 1
effect = 'ATT'
k = 1
print('\nATT k-nearest neighbour mathcing started with k=', k)
atthat_k1, sehat_att_k1, matchstat_att_k1 = mtch.matching_matching_est(effect=effect, ydata=y_sub_win, ddata=d_sub_win,
                                                                       xdata=prop_score_win, matchtype='knearest', k=k,
                                                                       distancefunction='L2cov', getSE=True,
                                                                       getmatchstats=True, matchstatsmode='mean_dist')
print('ATT estimation, k-nearest neighbour (k= %d): %.4f' % (k, atthat_k1))
print('ATT SE estimation, k-nearest neighbour (k= %d): %.4f' % (k, sehat_att_k1))
print('ATT matching statistics (k=', k ,')\n:', matchstat_att_k1)
# 5 nearest neighbour, k = 1
k = 5
print('\nATT k-nearest neighbour mathcing started with k=', k)
atthat_k2, sehat_att_k2, matchstat_att_k2 = mtch.matching_matching_est(effect=effect, ydata=y_sub_win, ddata=d_sub_win,
                                                                       xdata=prop_score_win, matchtype='knearest', k=k,
                                                                       distancefunction='L2cov',  getSE=True,
                                                                       getmatchstats=True, matchstatsmode='mean_dist')
print('ATT estimation, k-nearest neighbour (k= %d): %.4f' % (k, atthat_k2))
print('ATT SE estimation, k-nearest neighbour (k= %d): %.4f' % (k, sehat_att_k2))
print('ATT matching statistics (k=', k ,')\n:', matchstat_att_k2)

# Mathcing, ATE
# 1 nearest neighbour, k = 1
effect = 'ATE'
k = 1
print('\n#################\n ATE \n ##################')
print('\nATE k-nearest neighbour mathcing started with k=', k)
atehat_k1, sehat_ate_k1 = mtch.matching_matching_est(effect=effect, ydata=y_sub_win, ddata=d_sub_win,
                                                     xdata=prop_score_win, matchtype='knearest', k=k,
                                                     distancefunction='L2cov', getSE=True,
                                                     getmatchstats=False, matchstatsmode='mean_dist')
print('ATE estimation, k-nearest neighbour (k= %d): %.4f' %(k, atehat_k1))
print('ATE SE estimation, k-nearest neighbour (k= %d): %.4f' %(k, sehat_ate_k1))
# 5 nearest neighbour, k = 1
k = 5
print('\nATE k-nearest neighbour mathcing started with k=', k)
atehat_k2, sehat_ate_k2 = mtch.matching_matching_est(effect=effect, ydata=y_sub_win, ddata=d_sub_win,
                                                     xdata=prop_score_win, matchtype='knearest', k=k,
                                                     distancefunction='L2cov',  getSE=True,
                                                     getmatchstats=False, matchstatsmode='mean_dist')
print('ATE estimation, k-nearest neighbour (k= %d): %.4f' %(k, atehat_k2))
print('ATE SE estimation, k-nearest neighbour (k= %d): %.4f' %(k, sehat_ate_k2))
