# Advanced Econometrics 2 Problem set 1. Author: Mate Kormos

# Control the behavior of figures
savefigure = True
viewfigure = False

# Import dependencies
import numpy as np
import itertools
import scipy
from tabulate import tabulate
import advecon2ps1_functions as advecon2fun
import matplotlib as mpl
# Figure behaviour
if savefigure:
    # Scaling for LaTeX

    def figsize(scale):
            fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
            inches_per_pt = 1.0 / 72.27  # Convert pt to inch
            golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
            fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
            fig_height = fig_width * golden_mean  # height in inches
            fig_size = [fig_width, fig_height]
            return fig_size


    # Figure format settings
    pgf_with_rc_fonts = {'font.family': 'serif', 'figure.figsize': figsize(0.9), 'pgf.texsystem': 'pdflatex'}
mpl.rcParams.update(pgf_with_rc_fonts)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
plt.rc('font', family='serif')
# Close previous figures
plt.close()

# PROBLEM 1 #

# Part (c)
# Plotting the filter in frequency domain
# Compute the values
# Domain
omega = np.arange(0, np.pi, 0.05)
# Filter function in freq. domain


def ownfilter(w):
    return 1 / 9 * (3 + 4 * np.cos(w) + 2 * np.cos(2 * w))

# Compute the values
filtervalue = [ownfilter(w) for w in omega]
# Make the plot
f, ax = plt.subplots()
ax.plot(omega, filtervalue, label='$F(\omega)$', color='red', linewidth=2)
ax.legend()
ax.set_title('Smoothing filter in the frequency domain')
plt.setp(ax.get_xticklabels(), size=10, ha='center')
plt.setp(ax.get_yticklabels(), size=10)
plt.xlabel('$\omega$', size=12)
plt.ylabel('$F(\omega)$', size=12)
# Save figure if told so
if savefigure:
    f.set_size_inches(figsize(1.5)[0], figsize(0.9)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps1\Problem1_c.pgf', bbox_inches='tight')
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps1\Problem1_c.png', bbox_inches='tight')

# Show figure if reguired
if viewfigure:
    plt.show(block=viewfigure)


# PROBLEM 2 #

# Import data
with open(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps1\clothing.txt', 'r') as myfile:
    filein = myfile.readlines()
# Break lines and convert into numpy array
clothing = np.array([float(line.split()[0]) for line in filein])

# Part (a)
# Plotting the time series
# Time array
timeline = np.arange('1995-04', '2004-01', dtype='datetime64[M]').astype(datetime)
myfmt = mdates.DateFormatter('%Y-%m')
# Plot
f, ax = plt.subplots()
ax.plot_date(timeline, clothing, label='Clothing', color='red', linestyle='solid', linewidth=1.5)
ax.legend()
ax.set_title('Monthly inflation rate of clothing items')
ax.xaxis.set_major_formatter(myfmt)
plt.setp(ax.get_xticklabels(), size=10, ha='center')
plt.setp(ax.get_yticklabels(), size=10)
plt.xlabel('Time', size=12)
plt.ylabel('Inflation rate', size=12)
# Save figure if told so
if savefigure:
    f.set_size_inches(figsize(1.5)[0], figsize(0.9)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps1\Problem2_a1.pgf', bbox_inches='tight')
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps1\Problem2_a1.png', bbox_inches='tight')

# Show figure if reguired
if viewfigure:
    plt.show(block=viewfigure)

# Part (a)-(b)
# Domain
T = len(clothing)
omega1 = np.arange(2 * np.pi / T, (T - 1) / T * np.pi + np.finfo(float).eps, 2 * np.pi / T)
# Call spectral_density for the given frequencies and q=10, q=50
sdhat10 = [advecon2fun.spectral_density(w, clothing, usebartlett=True, q=10) for w in omega1]
sdhat50 = [advecon2fun.spectral_density(w, clothing, usebartlett=True, q=50) for w in omega1]
# Plot
f, ax = plt.subplots()
ax.plot(omega1, sdhat10, label='$\hat{S}(\omega)$, $q=10$', color='red', marker='*', ms=8, linewidth=2)
ax.plot(omega1, sdhat50, label='$\hat{S}(\omega)$, $q=50$', color='royalblue', marker='^', ms=8, linewidth=2)
ax.legend()
ax.set_title('Estimated spectral density of clothing inflation')
plt.setp(ax.get_xticklabels(), size=10, ha='center')
plt.setp(ax.get_yticklabels(), size=10)
plt.xlabel('$\omega$', size=12)
plt.ylabel('$\hat{S}(\omega)$', size=12)
# Save figure if told so
if savefigure:
    f.set_size_inches(figsize(1.5)[0], figsize(0.9)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps1\Problem2_ab.pgf', bbox_inches='tight')
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps1\Problem2_ab.png', bbox_inches='tight')

# Show figure if reguired
if viewfigure:
    plt.show(block=viewfigure)

# Part (c)
# Compute seasonal difference
clothing_d12 = advecon2fun.customdiff(clothing, 12)
clothing_d12_nonan = clothing_d12[12:]
# Plot
T = len(clothing) - 12
omega1 = np.arange(2 * np.pi / T, (T - 1) / T * np.pi + np.finfo(float).eps, 2 * np.pi / T)
# Call spectral_density for the given frequencies and q=10, q=50
sdhat10_d12 = [advecon2fun.spectral_density(w, clothing_d12_nonan, usebartlett=True, q=10) for w in omega1]
sdhat50_d12 = [advecon2fun.spectral_density(w, clothing_d12_nonan, usebartlett=True, q=50) for w in omega1]
# Plot
f, ax = plt.subplots()
ax.plot(omega1, sdhat10_d12, label='$\hat{S}(\omega)$, $q=10$', color='red', marker='*', ms=8, linewidth=2)
ax.plot(omega1, sdhat50_d12, label='$\hat{S}(\omega)$, $q=50$', color='royalblue', marker='^', ms=8, linewidth=2)
ax.legend()
ax.set_title('Estimated spectral density of seasonally diff. clothing inflation')
plt.setp(ax.get_xticklabels(), size=10, ha='center')
plt.setp(ax.get_yticklabels(), size=10)
plt.xlabel('$\omega$', size=12)
plt.ylabel('$\hat{S}(\omega)$', size=12)
# Save figure if told so
if savefigure:
    f.set_size_inches(figsize(1.5)[0], figsize(0.9)[1])
    # save to pgf
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps1\Problem2_c.pgf', bbox_inches='tight')
    # save to png
    plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Spring\AdvEcon2\AdvEcon2_ps1\Problem2_c.png', bbox_inches='tight')
# Show figure if reguired
if viewfigure:
    plt.show(block=viewfigure)


# Problem 6 #

# Y
y = np.array([2, 5, 3, 3])
# All possible resamples
resamples_tup = list(itertools.product([0,1], repeat=4))
# Convert tuples to arrays for calculation
resamples = [np.array(resamples_tup[i]) for i in range(0, len(resamples_tup))]
# For each sample calculate the  the probability of the sample and the test statistic, Y_treat-Y_control
arraytoprint = [(tuple(i), np.dot(y, i) / sum(i) - (sum(y) - np.dot(y, i)) / (4 - sum(i)),
                 (1 / 3) ** sum(i) * (2 / 3) ** (4-sum(i)))
                for i in resamples]# if (sum(i) != 0 and sum(i) != 4)]
print(tabulate(arraytoprint, tablefmt="latex", floatfmt=".4f"))
