# import libraries for analysis
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
from typing import Callable, Dict


# define constants
DATA_DIR = 'experiment2_analysis/data' # directory where the data is stored
OUTPUT_DIR = 'experiment2_analysis/out' # directory where the output will be saved
REPS = 30 # number of repetitions
LL_DELAYS = [5, 10, 20, 30, 40, 50, 60, 70, 80] # RIs used in the experiment
LL_MAG = 15 # FDF mean of the LL alternative


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


# define functions for the analysis
def estimate_indifference_point(mags: np.ndarray, props: np.ndarray) -> float:
    '''
    this function fits the sigmoid function to the magnitudes and proportions of choices for the LL alternative and estimates the indifference point by finding the x value where the sigmoid function is 0.5
    '''
    params, _ = curve_fit(sigmoid, mags, props, p0=[-0.9, 0.05, 100, 1], maxfev=1000000)

    estimation_x = np.linspace(min(mags), max(mags), 10000)
    estimation_y = sigmoid(estimation_x, *params)

    indiff_pt = estimation_x[np.argmin(np.abs(estimation_y - 0.5))]

    return indiff_pt

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    this function removes the first 500 generations of each schedule from the data
    '''
    
    cleaned_data = df.groupby(['File', 'Rep', 'Sched'])
    cleaned_data = cleaned_data.apply(lambda row: row.iloc[1:]).reset_index(drop=True)
    cleaned_data.sort_values(by=['SSMag', 'Rep', 'Sched'], inplace=True)

    return cleaned_data

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

def min_max_transform(indiff_pts: np.ndarray) -> np.ndarray:
    '''
    this function reverses the scale of the indifference points by subtracting each point from the sum of the minimum and maximum indifference points
    '''
    return ((min(indiff_pts) + max(indiff_pts)) - indiff_pts) 


# load in the raw data from the excel files in the data directory
files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]
dfs = []
for f in files:
    for rep in range(REPS):
        data = pd.read_excel(os.path.join(DATA_DIR, f), sheet_name=f"Repetition {rep+1}")
        # add columns for the repetition, file, SS magnitude, LL delay
        data['Rep'] = rep + 1
        data['File'] = f
        data['SSMag'] = int(f.split('.')[0].replace('ssmag', ''))
        data['LLDelay'] = data['Sched'].apply(lambda x: int(LL_DELAYS[x-1]))

        dfs.append(data)

# concatenate the dataframes to create a single dataframe with all the raw data
raw_data = pd.concat(dfs)


# clean the data by removing the acquisition periods (first 500 generations)
cleaned_data = clean_data(raw_data)


# get the mean response rate and reinforcer rate for each schedule for each repetition
ind_ao_data = cleaned_data.groupby(['File', 'Rep', 'Sched']).mean().reset_index()

ind_ao_data['PropSS'] = ind_ao_data['B1'] / (ind_ao_data['B1'] + ind_ao_data['B2']) # calculate the proportion of choices for the SS alternative


# get the mean and standard deviation of the response rate and reinforcer rate for each schedule across all repetitions
mean_sd_data = cleaned_data.groupby(['File', 'Sched']).agg(['mean', 'std']).reset_index()

mean_sd_data['PropSS'] = mean_sd_data[('B1', 'mean')] / (mean_sd_data[('B1', 'mean')] + mean_sd_data[('B2', 'mean')]) # calculate the mean proportion of choices for the SS alternative


# estimate the indifference points

# individual AO indifference points
ind_indiff_pts = {
    'Rep': [],
    'LLDelay': [],
    'IndiffPt': []
}
# estimate the indifference points for each repetition (AO) and delay
for rep in ind_ao_data['Rep'].unique():
    for delay in ind_ao_data['LLDelay'].unique():
        ind_data = ind_ao_data[(ind_ao_data['Rep'] == rep) & (ind_ao_data['LLDelay'] == delay)]
        indiff_pt = estimate_indifference_point(ind_data['SSMag'].values, ind_data['PropSS'].values)
        ind_indiff_pts['Rep'].append(rep)
        ind_indiff_pts['LLDelay'].append(delay)
        ind_indiff_pts['IndiffPt'].append(indiff_pt)

# convert the dictionary to a dataframe
ind_indiff_pts = pd.DataFrame(ind_indiff_pts)


# estimate the indifference points for the mean data
mean_indiff_pts = {
    'LLDelay': [],
    'IndiffPt': []
}
# estimate the indifference points for each delay
for ll_delay in mean_sd_data[('LLDelay', 'mean')].unique():
    mean_data = mean_sd_data[mean_sd_data[('LLDelay', 'mean')] == ll_delay]
    indiff_pt = estimate_indifference_point(mean_data[('SSMag', 'mean')].values, mean_data['PropSS'].values)
    mean_indiff_pts['LLDelay'].append(ll_delay)
    mean_indiff_pts['IndiffPt'].append(indiff_pt)

# convert the dictionary to a dataframe
mean_indiff_pts = pd.DataFrame(mean_indiff_pts)


# fit the discounting models to the individual AO indifference points
# create a dictionary to store the fits of the discounting models
ind_fits_dict = {
    "Rep": [],
    "Mazur k": [],
    "Mazur R^2": [],
    "Mazur Sy.x": [],
    "Mazur Residuals": [],
    "Myerson & Green k": [],
    "Myerson & Green s": [],
    "Myerson & Green R^2": [],
    "Myerson & Green Sy.x": [],
    "Myerson & Green Residuals": [],
    "Rachlin k": [],
    "Rachlin s": [],
    "Rachlin R^2": [],
    "Rachlin Sy.x": [],
    "Rachlin Residuals": []
}

for rep in ind_indiff_pts['Rep'].unique():
    rep_data = ind_indiff_pts[ind_indiff_pts['Rep'] == rep]

    # get the x and y values for the fitting
    # the LL delay of 5 is removed due to prevalence of exclusive preference
    x = rep_data['LLDelay'].values[1:]
    y = np.array(min_max_transform(rep_data['IndiffPt'].values[1:]))
    
    # calculate A for this AO
    A = min(y) + max(y) - LL_MAG

    # fit the Mazur model
    mazur_fit = fit_model(x, y, mazur, p0=[0.05])

    # fit the Myerson & Green model
    myerson_green_fit = fit_model(x, y, myerson_green, p0=[0.05, 1])

    # fit the Rachlin model
    rachlin_fit = fit_model(x, y, rachlin, p0=[0.05, 1])

    # add the parameters to the dictionary
    ind_fits_dict['Rep'].append(rep)
    ind_fits_dict['Mazur k'].append(mazur_fit['params'][0])
    ind_fits_dict['Mazur R^2'].append(mazur_fit['r_squared'])
    ind_fits_dict['Mazur Sy.x'].append(mazur_fit['sy_x'])
    ind_fits_dict['Mazur Residuals'].append(mazur_fit['residuals'])
    ind_fits_dict['Myerson & Green k'].append(myerson_green_fit['params'][0])
    ind_fits_dict['Myerson & Green s'].append(myerson_green_fit['params'][1])
    ind_fits_dict['Myerson & Green R^2'].append(myerson_green_fit['r_squared'])
    ind_fits_dict['Myerson & Green Sy.x'].append(myerson_green_fit['sy_x'])
    ind_fits_dict['Myerson & Green Residuals'].append(myerson_green_fit['residuals'])
    ind_fits_dict['Rachlin k'].append(rachlin_fit['params'][0])
    ind_fits_dict['Rachlin s'].append(rachlin_fit['params'][1])
    ind_fits_dict['Rachlin R^2'].append(rachlin_fit['r_squared'])
    ind_fits_dict['Rachlin Sy.x'].append(rachlin_fit['sy_x'])
    ind_fits_dict['Rachlin Residuals'].append(rachlin_fit['residuals'])

# convert the dictionary to a dataframe
ind_fits = pd.DataFrame(ind_fits_dict)



# fit the discounting models to the indifference points estimated from the mean data
mean_fits_dict = {
    "Mazur k": [],
    "Mazur R^2": [],
    "Mazur Sy.x": [],
    "Mazur Residuals": [],
    "Myerson & Green k": [],
    "Myerson & Green s": [],
    "Myerson & Green R^2": [],
    "Myerson & Green Sy.x": [],
    "Myerson & Green Residuals": [],
    "Rachlin k": [],
    "Rachlin s": [],
    "Rachlin R^2": [],
    "Rachlin Sy.x": [],
    "Rachlin Residuals": []
}

# get the x and y values for the fitting
# the LL delay of 5 is removed due to prevalence of exclusive preference
x = mean_indiff_pts['LLDelay'].values[1:]
y = np.array(min_max_transform(mean_indiff_pts['IndiffPt'].values[1:]))

# calculate A for the mean data
A = min(y) + max(y) - LL_MAG

# fit the Mazur model
mazur_fit = fit_model(x, y, mazur, p0=[0.05])

# fit the Myerson & Green model
myerson_green_fit = fit_model(x, y, myerson_green, p0=[0.05, 1])

# fit the Rachlin model
rachlin_fit = fit_model(x, y, rachlin, p0=[0.05, 1])

# add the parameters to the dictionary
mean_fits_dict['Mazur k'].append(mazur_fit['params'][0])
mean_fits_dict['Mazur R^2'].append(mazur_fit['r_squared'])
mean_fits_dict['Mazur Sy.x'].append(mazur_fit['sy_x'])
mean_fits_dict['Mazur Residuals'].append(mazur_fit['residuals'])
mean_fits_dict['Myerson & Green k'].append(myerson_green_fit['params'][0])
mean_fits_dict['Myerson & Green s'].append(myerson_green_fit['params'][1])
mean_fits_dict['Myerson & Green R^2'].append(myerson_green_fit['r_squared'])
mean_fits_dict['Myerson & Green Sy.x'].append(myerson_green_fit['sy_x'])
mean_fits_dict['Myerson & Green Residuals'].append(myerson_green_fit['residuals'])
mean_fits_dict['Rachlin k'].append(rachlin_fit['params'][0])
mean_fits_dict['Rachlin s'].append(rachlin_fit['params'][1])
mean_fits_dict['Rachlin R^2'].append(rachlin_fit['r_squared'])
mean_fits_dict['Rachlin Sy.x'].append(rachlin_fit['sy_x'])
mean_fits_dict['Rachlin Residuals'].append(rachlin_fit['residuals'])

# convert the dictionary to a dataframe
mean_fits = pd.DataFrame(mean_fits_dict)


# output the results to csv files
ind_ao_data.to_csv(f'{OUTPUT_DIR}/ind_ao_data.csv', index=False)
mean_sd_data.to_csv(f'{OUTPUT_DIR}/mean_sd_data.csv', index=False)
ind_indiff_pts.to_csv(f'{OUTPUT_DIR}/ind_indiff_pts.csv', index=False)
mean_indiff_pts.to_csv(f'{OUTPUT_DIR}/mean_indiff_pts.csv', index=False)
ind_fits.to_csv(f'{OUTPUT_DIR}/ind_fits.csv', index=False)
mean_fits.to_csv(f'{OUTPUT_DIR}/mean_fits.csv', index=False)

    



