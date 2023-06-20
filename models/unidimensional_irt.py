"""
Methods for Model objects where the model type is UIRT.

@author: Noah Burrell <burrelln@umich.edu>

"""
import numpy as np
from scipy.special import expit

import girth

from .format_df import girth_format

    
def compute_probability(ability, difficulty, discrimination, guessing):
    
    exponent = discrimination*(ability - difficulty)
    quotient = expit(exponent)
    prob = guessing + (1 - guessing)*quotient
    
    return prob

def replace_nan(abilities):
    
    avg = np.nanmean(abilities)
    replaced = np.nan_to_num(abilities, nan=avg)
    
    return replaced

def fit_1PL(df):
    
    ir_input, wid_dict, tid_dict = girth_format(df)
    
    results = girth.onepl_mml(ir_input, options={'estimate_distribution': True, 'number_of_samples': 10})
    difficulties = results["Difficulty"]
    discrimination_val = results["Discrimination"]
    discrimination = np.full_like(difficulties, discrimination_val)
    guessing = np.full_like(difficulties, 0)
    
    abilities_1PL = girth.ability_mle(ir_input, difficulties, discrimination)
    abilities = replace_nan(abilities_1PL)

    return wid_dict, tid_dict, abilities, difficulties, discrimination, guessing

def fit_2PL(df):
    
    ir_input, wid_dict, tid_dict = girth_format(df)
    
    results = girth.twopl_mml(ir_input, options={'estimate_distribution': True, 'number_of_samples': 10})
    difficulties = results["Difficulty"]
    discrimination = results["Discrimination"]
    if len(discrimination.shape) > 1:
        discrimination = discrimination[:, 0]
    guessing = np.full_like(difficulties, 0)
    
    abilities_2PL = girth.ability_mle(ir_input, difficulties, discrimination)
    abilities = replace_nan(abilities_2PL)

    return wid_dict, tid_dict, abilities, difficulties, discrimination, guessing

def fit_3PL(df):
    
    ir_input, wid_dict, tid_dict = girth_format(df)
    
    results = girth.threepl_mml(ir_input, options={'estimate_distribution': True, 'number_of_samples': 10})
    difficulties = results["Difficulty"]
    discrimination = results["Discrimination"]
    discrimination = results["Discrimination"]
    guessing = results["Guessing"]
    
    abilities_3PL = girth.ability_3pl_mle(ir_input, difficulties, discrimination, guessing)
    abilities = replace_nan(abilities_3PL)

    return wid_dict, tid_dict, abilities, difficulties, discrimination, guessing
    
def compute_log_likelihood(response_array, abilities, difficulties, discrimination, guessing):
    
    difficulty_column = difficulties.reshape(-1, 1)
    discrimination_column = discrimination.reshape(-1, 1)
    guessing_column = guessing.reshape(-1, 1)
    prob_matrix = compute_probability(abilities, difficulty_column, discrimination_column, guessing_column)
    density_vals_correct = prob_matrix[response_array == 1]
    density_vals_incorrect = (1 - prob_matrix[response_array == 0])
    log_likelihood = np.log(density_vals_correct).sum() + np.log(density_vals_incorrect).sum()
    
    return log_likelihood

def compute_aic_bic(response_array, abilities, difficulties, discrimination, guessing, model_type):
    
    llh = compute_log_likelihood(response_array, abilities, difficulties, discrimination, guessing)
    if model_type == "1PL":
        params = abilities.size + difficulties.size + 1
    elif model_type == "2PL":
        params = abilities.size + difficulties.size + discrimination.size
    elif model_type == "3PL":
        params = abilities.size + difficulties.size + discrimination.size + guessing.size
    else:
        print("Invalid model type.")
    n = np.count_nonzero(response_array != girth.INVALID_RESPONSE)
    aic = 2 * (params - llh)
    bic = (params * np.log(n) - 2 * llh)
    
    return aic, bic