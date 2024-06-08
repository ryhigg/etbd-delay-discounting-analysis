# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Callable, Dict

plt.rcParams.update({
'figure.figsize': (6, 6),
'figure.dpi': 300,
'axes.spines.top': False,
'axes.spines.right': False,
'axes.linewidth': 1.0,
'axes.prop_cycle': plt.cycler(color='kbrgymc'),
'font.family': 'Helvetica Neue',
'font.size': 12,
'legend.frameon': 'false'
})

# %%
# define equations used in the analysis
def sigmoid(x, a, b, c, d):
    '''
    this is used to estimate the indifference points
    '''
    return a / (1 + np.exp(-(b * (x - c))))+d

def cubic(x, a, b, c, d):
    '''
    this is used for the cubic trend test (McDowell et al., 2016) to evaluate residuals 
    '''
    return a*x**3 + b*x**2 + c*x + d

# discounting models
# note: A must be defined before using these functions
def mazur(x, k):
    return A / (1 + k*x)

def rachlin(x, k, s):
    return A / (1 + k*x**s)

def myerson_green(x, k, s):
    return A / (1 + k*x)**s

# %%
# define functions for fitting models
def fit_model(x: np.ndarray, y: np.ndarray, model: Callable, p0: list) -> Dict:
    '''
    this function fits a model to the data and returns the parameters, r-squared, Sy.x, and residuals
    '''
    params, cov = curve_fit(model, x, y, p0=p0, maxfev=1000)

    residuals = y - model(x, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    sy_x = np.sqrt(ss_res / (len(y) - 1))

    return {
        'params': params,
        'r_squared': r_squared,
        'sy_x': sy_x,
        'residuals': residuals
    }



# %%
# load the data
ind_ao_data = pd.read_csv('out/ind_ao_data.csv')



# %%
a027_data = ind_ao_data[ind_ao_data['Rep'] == 27]

fig, axs = plt.subplots(2, 4, figsize=(12, 6))
axs = axs.flatten()

for i, ll_delay in enumerate(a027_data['LLDelay'].unique()):
    if ll_delay == 5: # skip delay 5 due to prevelance of exclusive preference
        continue
    data = a027_data[a027_data['LLDelay'] == ll_delay]
    ax = axs[i-1]

    ssmags = data['SSMag'].values
    prop_ss = data['PropSS'].values

    popt, pcov = curve_fit(sigmoid, ssmags, prop_ss, p0=[-0.9, 0.05, 100, 1], maxfev=10000)

    plotting_x = np.linspace(min(ssmags), max(ssmags), 10000)
    plotting_y = sigmoid(plotting_x, *popt)

    indiff_pt = plotting_x[np.argmin(np.abs(plotting_y - 0.5))]

    res = prop_ss - sigmoid(ssmags, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((prop_ss - np.mean(prop_ss))**2)
    r_squared = 1 - (ss_res / ss_tot)

    ax.scatter(ssmags, prop_ss, facecolors='none', edgecolors='k')
    ax.plot(plotting_x, plotting_y, label=f'$R^2$ = {r_squared:.2f}')
    ax.axvline(indiff_pt, linestyle='--', alpha=0.5, label=f'IP: {indiff_pt:.2f}')
    ax.axhline(0.5, linestyle='--', alpha=0.5)
    ax.set_title(f'Larger Later Delay: {ll_delay}')
    ax.legend()

fig.tight_layout()
fig.text(0.5, -0.01, 'Smaller Sooner Magnitude', ha='center', fontsize=16)
fig.text(-0.01, 0.5, 'Proportion of Smaller Sooner Choices', va='center', rotation='vertical', fontsize=16)
plt.savefig('out/ao_27_estimation.png')
plt.show()



    
# %%


