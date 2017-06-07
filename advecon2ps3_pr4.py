"""
Advanced Econometrics 2 Problem set 3 Problem 4. Author: Mate Kormos

"""
# import dependencies
import numpy as np
import os
import robustse

# looping through data files
for filename in ['group_n100_G10.txt', 'group_n100_G50.txt', 'group_n300_G50.txt']:
    print('\n###############################\n Work has started on file:', filename, '\n#########################\n')
    # get the data
    with open(os.path.join(os.getcwd(), filename), 'r') as myfile:
        filein = myfile.readlines()
    # break the lines, convert into np array
    # y
    y = np.array([float(line.split()[0]) for line in filein])
    # x
    x = np.array([float(line.split()[1]) for line in filein])
    # size
    n = len(y)

    # obtain the OLS point estimates
    # generate and append constant to create desing matrix
    constant = np.ones((n, 1))
    # design matrix
    X = np.concatenate((constant, x[:, None]), axis=1)
    # betahat
    XTXinv = np.linalg.inv((X.T).dot(X))
    bhat_p1 = XTXinv.dot(X.T)
    bhat = bhat_p1.dot(y)
    # usual cov matrix
    uhat = y - X.dot(bhat)
    s2hat = (uhat.T).dot(uhat) / (n - 2)
    covhat = s2hat * XTXinv

    # obtain the robust covariannce matrix
    # create group ids
    if filename == 'group_n100_G10.txt':
        # number of observations per group
        n_g = 100
        # number of groups
        G = 10
    elif filename == 'group_n100_G50.txt':
        # number of observations per group
        n_g = 100
        # number of groups
        G = 50
    elif filename == 'group_n300_G50.txt':
        # number of observations per group
        n_g = 300
        # number of groups
        G = 50
    group_id=np.empty((n_g * G, ))
    for g in range(G):
        group_id[g* n_g: (g + 1) * n_g] = g
    # obtain the estimate
    bootsamplenumber = 1000
    covhat_rob = robustse.robustse_bootstrap(xdata=X, ydata=y, groupid=group_id, bootsamplenumber=bootsamplenumber)

    # print
    print('For file ', filename, 'the point estimate is \n', bhat, '\n\nthe usual cov matrix is \n', covhat,
          '\n the robust is\n', covhat_rob)
