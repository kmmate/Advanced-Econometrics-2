"""
Advanced Econometrics 2 Problem set 3 Problem 1. Author: Mate Kormos

Note: prior to runnig change the xlsx data to txt setting the delimeters to points instead of commas, make sure
 all the decimals are displeyed when saved as txt in Excel
"""
# import dependencies
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl

# figure behaviour
savefigure = True
if savefigure:
    # scaling for LaTeX

    def figsize(scale):
            fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
            inches_per_pt = 1.0 / 72.27  # Convert pt to inch
            golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
            fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
            fig_height = fig_width * golden_mean  # height in inches
            fig_size = [fig_width, fig_height]
            return fig_size


    # figure format settings
    pgf_with_rc_fonts = {'font.family': 'serif', 'figure.figsize': figsize(0.9), 'pgf.texsystem': 'pdflatex'}
mpl.rcParams.update(pgf_with_rc_fonts)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
plt.rc('font', family='serif')
# close previous figures
plt.close()

# get the data
with open(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps3\discontinuity.txt', 'r')\
          as myfile:
    filein = myfile.readlines()
# break the lines, cut the header, and convert into np array
# y
rawdata = [line.split()[0] for line in filein][1:]
y = np.array([float(i) for i in rawdata])
# x
rawdata = [line.split()[1] for line in filein][1:]
x = np.array([float(i) for i in rawdata])
# generate d
n = 1000
d = np.zeros((n,))
d[x > 5] = 1
print('Number of treated individuals: %d' % d.sum())

########################################## PART A ###################################
print('################################################\n PART A\n ############################################\n')
# plotting
f, ax = plt.subplots()
ax.plot(x[x < 5], y[x < 5], '.', label='Control', color='blue', linewidth=2)
ax.plot(x[x > 5], y[x > 5], '.', label='Treated', color='red', linewidth=2)
ax.legend()
ax.set_title('Scatter plot $(x, y)$')
plt.setp(ax.get_xticklabels(), size=10, ha='center')
plt.setp(ax.get_yticklabels(), size=10)
plt.xlabel('$x$', size=12)
plt.ylabel('$y$', size=12)
# save figure if told so
if savefigure:
    f.set_size_inches(figsize(1.5)[0], figsize(0.9)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps3\Problem1_a.pgf', bbox_inches='tight')
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps3\Problem1_a.png', bbox_inches='tight')
plt.show(block=False)
# regression
# create design matrix
constant = np.ones((n, ))
# design matrix
X = np.concatenate(( d[:, None], constant[:, None]), axis=1)
# regression
results = sm.OLS(endog=y, exog=X, hasconst=True).fit()
#print(results.summary())

########################################## PART B ##########################################
print('################################################\n PART B\n ############################################\n')
# create all the variables
# x's
xshift = x - 5
xshift_sq = xshift ** 2
# interaction
xshift_d = xshift * d
xshift_sq_d = xshift_sq * d

# design matrix
X_b = np.concatenate((d[:, None], constant[:, None]), axis=1)

# restrict the sample
mask1 = np.all([(x >= 4), (x <= 6)], axis=0)
mask2 = np.all([(x >= 4.75), (x <= 5.25)], axis=0)
# run OLS with mask1
results = sm.OLS(endog=y[mask1], exog=X_b[mask1], hasconst=True).fit()
print('Regression results for observations {i: x_i in [4, 6]}\n', results.summary())
# run OLS with mask2
results = sm.OLS(endog=y[mask2], exog=X_b[mask2], hasconst=True).fit()
print('Regression results for observations {i: x_i in [4.75, 5.25]}\n', results.summary())

######################################### PART C ###################################################
print('################################################\n PART C\n ############################################\n')
# design matrix
X_c = np.concatenate((d[:, None], xshift[:, None], xshift_d[:, None], constant[:, None]), axis=1)
results = sm.OLS(endog=y, exog=X_c, hasconst=True).fit()
#print(results.summary())

######################################### PART D ###################################################
print('################################################\n PART D\n ############################################\n')
# design matrix
X_d = np.concatenate((d[:, None], xshift[:, None], xshift_sq[:, None], xshift_d[:, None], xshift_sq_d[:, None],
                      constant[:, None]), axis=1)
results = sm.OLS(endog=y, exog=X_d, hasconst=True).fit()
#print(results.summary())

######################################### PART E ###################################################
print('################################################\n PART E\n ############################################\n')
# design matrix
X_e = np.concatenate((d[:, None], xshift[:, None], xshift_d[:, None], constant[:, None]), axis=1)
# run OLS with mask1
results = sm.OLS(endog=y[mask1], exog=X_e[mask1], hasconst=True).fit()
print('Regression results for observations {i: x_i in [4, 6]}\n', results.summary())
# run OLS with mask 2
results = sm.OLS(endog=y[mask2], exog=X_e[mask2], hasconst=True).fit()
print('Regression results for observations {i: x_i in [4.75, 5.25]}\n', results.summary())

