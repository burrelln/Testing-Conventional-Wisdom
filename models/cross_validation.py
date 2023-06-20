"""
Code for k-fold cross validation.

@author: Noah Burrell <burrelln@umich.edu>

"""
import numpy as np

from .model import Model


def quad_scoring_rule(prediction, outcome):
    distribution = (1-prediction, prediction)
    score = 2*distribution[outcome] - np.dot(distribution, distribution)
    return score

def kfold_cross_validation(k, df, model_type, model_subtype, rng):
    
    indices = np.array(df.index)
    
    rng.shuffle(indices)
    
    scores = []
    
    i = 0
    while i < len(indices):
        print(i)
        j = i + k
        leave_out = indices[i:j]
        
        kept_df = df.drop(leave_out)
        kept_df = kept_df.reset_index(drop=True)
        m = Model(model_type, model_subtype)
        m.fit(kept_df)
        
        left_out_df = df.loc[leave_out]
        for index, df_row in left_out_df.iterrows():
            wid = df_row["user_id"]
            tid = df_row["task_id"]
            r = df_row["report"]
            gt = df_row["ground_truth"]
            
            prob_correct = m.predict(wid, tid)
            correct = int(r == gt)
            
            if not prob_correct:
                continue
            
            scores.append(quad_scoring_rule(prob_correct, correct))
            
        i = j
        
    return np.sum(scores)