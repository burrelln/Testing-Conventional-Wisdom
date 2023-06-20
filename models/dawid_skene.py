"""
Methods for Model objects where the model type is DS.

@author: Noah Burrell <burrelln@umich.edu>

"""
import numpy as np


def compute_probability(ground_truth, confusion_matrix):
    
    prob = confusion_matrix[ground_truth, ground_truth]
    
    return prob

def fit_MLE(df):
    
    confusion_matrices = {}
    
    task_df = df[["task_id", "ground_truth"]].groupby(["task_id"]).max()
    num_tasks = len(task_df.index)
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    
    user_df = df.groupby(["user_id", "ground_truth"]).mean()
    avg0 = user_df["correct"].xs(0, level="ground_truth").mean()
    avg1 = user_df["correct"].xs(1, level="ground_truth").mean()
    
    for wid, w_df in user_df.groupby(level="user_id"):
        vals = [avg0, avg1]
        for gt in w_df.index.get_level_values("ground_truth"):
            vals[gt] = w_df.loc[(wid, gt)]["correct"]
           
        p, q = vals    
        
        # Hedge 0 and 1 values because of LLH computations
        if p in [0, 1]:
            p = (1/num_tasks)*0.5 + ((num_tasks-1)/num_tasks)*p
        if q in [0, 1]:
            q = (1/num_tasks)*0.5 + ((num_tasks-1)/num_tasks)*q
        
        cm = np.array([
                        [p, 1 - p], 
                        [1 - q, q]
                    ])
        confusion_matrices[wid] = cm
           
    return confusion_matrices

def compute_log_likelihood(df, confusion_matrices):
    
    count_dict = {wid:np.zeros_like(cm) for wid, cm in confusion_matrices.items()}
    
    for index, df_row in df.iterrows():
        
        
        wid = df_row["user_id"]
        r = df_row["report"]
        gt = df_row["ground_truth"]
        
        count_array = count_dict[wid]

        r = df_row["report"]
        gt = df_row["ground_truth"]
        
        count_array[gt, r] += 1
        
    response_log_likelihoods = np.array([count_dict[wid]*np.log(confusion_matrices[wid]) for wid in confusion_matrices.keys()])
    log_likelihood = response_log_likelihoods.sum()
    
    return log_likelihood

def compute_aic_bic(df, confusion_matrices, model_type="DS"):
    
    llh = compute_log_likelihood(df, confusion_matrices)
    if model_type == "DS":
        params = 2*len(confusion_matrices)
    else:
        print("Invalid model type.")
    n = len(df.index)
    aic = 2 * (params - llh)
    bic = (params * np.log(n) - 2 * llh)
    
    return aic, bic