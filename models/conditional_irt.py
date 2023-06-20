"""
Methods for Model objects where the model type is CIRT.

@author: Noah Burrell <burrelln@umich.edu>

"""
import numpy as np

from .format_df import girth_format
from . import unidimensional_irt as UIRT


def compute_probability(ability, difficulty, discrimination, guessing):
    
    prob = UIRT.compute_probability(ability, difficulty, discrimination, guessing)
    
    return prob

def get_dfs_by_gt(df):
    
    df0 = df.loc[df['ground_truth'].eq(0)].copy()
    df1 = df.loc[df['ground_truth'].eq(1)].copy()
    
    return df0, df1

def fit_conditional_1PL(df):
    
    df0, df1 = get_dfs_by_gt(df)
    
    wid_dict0, tid_dict0, abilities0, difficulties0, discrimination0, guessing0 = UIRT.fit_1PL(df0)
    wid_dict1, tid_dict1, abilities1, difficulties1, discrimination1, guessing1 = UIRT.fit_1PL(df1)
    
    param_dict = {0: [wid_dict0, tid_dict0, abilities0, difficulties0, discrimination0, guessing0], 1: [wid_dict1, tid_dict1, abilities1, difficulties1, discrimination1, guessing1]}
    
    return param_dict

def fit_conditional_2PL(df):
    
    df0, df1 = get_dfs_by_gt(df)
    
    wid_dict0, tid_dict0, abilities0, difficulties0, discrimination0, guessing0 = UIRT.fit_2PL(df0)
    wid_dict1, tid_dict1, abilities1, difficulties1, discrimination1, guessing1 = UIRT.fit_2PL(df1)
    
    param_dict = {0: [wid_dict0, tid_dict0, abilities0, difficulties0, discrimination0, guessing0], 1: [wid_dict1, tid_dict1, abilities1, difficulties1, discrimination1, guessing1]}
    
    return param_dict

def fit_conditional_3PL(df):
    
    df0, df1 = get_dfs_by_gt(df)
    
    wid_dict0, tid_dict0, abilities0, difficulties0, discrimination0, guessing0 = UIRT.fit_3PL(df0)
    wid_dict1, tid_dict1, abilities1, difficulties1, discrimination1, guessing1 = UIRT.fit_3PL(df1)
    
    param_dict = {0: [wid_dict0, tid_dict0, abilities0, difficulties0, discrimination0, guessing0], 1: [wid_dict1, tid_dict1, abilities1, difficulties1, discrimination1, guessing1]}
    
    return param_dict
    
def compute_log_likelihood(response_df, param_dict):
    
    rdf0, rdf1 = get_dfs_by_gt(response_df)
    
    ra0 = girth_format(rdf0)
    ra1 = girth_format(rdf1)
    
    llh0 = UIRT.compute_log_likelihood(ra0, param_dict[0][-4], param_dict[0][-3], param_dict[0][-2], param_dict[0][-1])
    llh1 = UIRT.compute_log_likelihood(ra1, param_dict[1][-4], param_dict[1][-3], param_dict[1][-2], param_dict[1][-1])
    
    log_likelihood = llh0 + llh1
    
    return log_likelihood

def compute_aic_bic(response_df, param_dict, model_type):
    
    llh = compute_log_likelihood(response_df, param_dict)
    if model_type == "1PL":
        params = (param_dict[0][-4].size + param_dict[1][-4].size) + (param_dict[0][-3].size + param_dict[1][-3].size) + 2
    elif model_type == "2PL":
        params = (param_dict[0][-4].size + param_dict[1][-4].size) + (param_dict[0][-3].size + param_dict[1][-3].size) + (param_dict[0][-2].size + param_dict[1][-2].size)
    elif model_type == "3PL":
        params = (param_dict[0][-4].size + param_dict[1][-4].size) + (param_dict[0][-3].size + param_dict[1][-3].size) + (param_dict[0][-2].size + param_dict[1][-2].size) + (param_dict[0][-1].size + param_dict[1][-1].size)
    else:
        print("Invalid model type.")
        
    n = len(response_df.index)
    
    aic = 2 * (params - llh)
    bic = (params * np.log(n) - 2 * llh)
    
    return aic, bic