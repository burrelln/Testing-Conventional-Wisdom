"""
Helper functions for loading and formatting data.

@author: Noah Burrell <burrelln@umich.edu>

"""
import numpy as np

import girth


def girth_format(df):
    """
    Transforms data from a standard DataFrame format to the item-response matrix needed for IRT computations with the girth package.

    Parameters
    ----------
    df : pd.DataFrame
        Standard DF format across all datasets, with (at least) the following columns:
            - "user_id": unique val corresponding to individuals
            - "task_id": unique val corresponding to tasks
            - "report": report provided by individual "user_id" on task "task_id"
            - "ground_truth": correct answer on task "task_id"

    Returns
    -------
    girth_input : 2-D np.array
        entry at [i, j] == 1 if task i was completed correctly by worker j
        entry at [i, j] == 0 if task i was completed incorrectly by worker j
        entry at [i, j] == girth.INVALID_RESPONSE if task i was not completed by worker j (indicates missing value)

    """
    
    num_unique = df.nunique()
    num_tasks = num_unique["task_id"]
    num_workers = num_unique["user_id"]
    
    ir_mtx = np.empty((num_tasks, num_workers))
    ir_mtx.fill(girth.INVALID_RESPONSE)
    
    tid_dict = {}
    wid_dict = {}
    
    tid_counter = 0
    wid_counter = 0
    
    for index, df_row in df.iterrows():
        wid = df_row["user_id"]
        tid = df_row["task_id"]
        r = df_row["report"]
        gt = df_row["ground_truth"]
        
        if tid not in tid_dict.keys():
            tid_dict[tid] = tid_counter
            tid_counter += 1
        
        row_idx = tid_dict[tid]
        
        if wid not in wid_dict.keys():
            wid_dict[wid] = wid_counter
            wid_counter += 1
        
        column_idx = wid_dict[wid]
        
        if r in (0, 1):
            entry = int(r == gt)
            ir_mtx[row_idx, column_idx] = entry
    
    girth_input = ir_mtx.astype(int)
    
    return girth_input, wid_dict, tid_dict