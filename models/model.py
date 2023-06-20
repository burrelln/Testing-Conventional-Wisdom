"""
Abstract Model class to provide one interface for all of the models that we consider.

@author: Noah Burrell <burrelln@umich.edu>

"""
import numpy as np

from .format_df import girth_format

from . import dawid_skene as DS
from . import conditional_irt as CIRT
from . import unidimensional_irt as UIRT


class Model:
    
    models = {
            0: "Dawid-Skene",
            1: "UIRT",
            3: "CIRT",
        }
    
    subtypes = {
            -1: "MLE", # DS only
            1: "1PL",  # IRT only 
            2: "2PL",  # IRT only
            3: "3PL"   # IRT only
        }

    
    def __init__(self, model, subtype=0):
        
        self.model_type = self.models[model]
        self.model_subtype = self.subtypes[subtype]
        self.workers = {}
        self.tasks = {}
        self.ability_means = {}
    
    def fit(self, response_df):
        
        if self.model_type == "Dawid-Skene":
            subtype = self.model_subtype
            if subtype == "MLE":
                self.workers = DS.fit_MLE(response_df)
            elif subtype == "EM":
                self.workers = DS.fit_EM(response_df)
            task_df = response_df[["task_id", "ground_truth"]].groupby(["task_id"]).max()
            self.tasks = {tup[0]:tup[1] for tup in task_df.itertuples()}
            
        elif self.model_type == "UIRT":
            subtype = self.model_subtype
            if subtype == "3PL":
                wid_dict, tid_dict, abilities, difficulties, discrimination, guessing = UIRT.fit_3PL(response_df)
            elif subtype == "2PL":
                wid_dict, tid_dict, abilities, difficulties, discrimination, guessing = UIRT.fit_2PL(response_df)
            else:
                wid_dict, tid_dict, abilities, difficulties, discrimination, guessing = UIRT.fit_1PL(response_df)
            for wid, idx in wid_dict.items():
                self.workers[wid] = abilities[idx]
            for tid, idx in tid_dict.items():
                self.tasks[tid] = (difficulties[idx], discrimination[idx], guessing[idx])
                
        elif self.model_type == "CIRT":
            subtype = self.model_subtype
            if subtype == "3PL":
                param_dict = CIRT.fit_conditional_3PL(response_df)
            elif subtype == "2PL":
                param_dict = CIRT.fit_conditional_2PL(response_df)
            else:
                param_dict = CIRT.fit_conditional_1PL(response_df)
                
            ability_means = {}
                
            for gt, params in param_dict.items():
                wid_dict = params[0]
                tid_dict = params[1]
                abilities = params[2]
                difficulties = params[3]
                discrimination = params[4]
                guessing = params[5]
                
                ability_means[gt] = np.mean(abilities)
                
                for wid, idx in wid_dict.items():
                    if wid not in self.workers.keys():
                        self.workers[wid] = {}
                    self.workers[wid][gt] = abilities[idx]
                    
                for tid, idx in tid_dict.items():
                    self.tasks[tid] = (difficulties[idx], discrimination[idx], guessing[idx], gt)
                    
            self.ability_means = ability_means
                    
            for wid, ability_dict in self.workers.items():
                for gt in [0, 1]:
                    if gt not in ability_dict.keys():
                        ability_dict[gt] = ability_means[gt]
                
    def predict(self, worker_id, task_id):
        
        if worker_id not in self.workers.keys() or task_id not in self.tasks.keys():
            prob = None
            
        else:
        
            if self.model_type == "Dawid-Skene":
                cm = self.workers[worker_id]
                gt = self.tasks[task_id]
                prob = DS.compute_probability(gt, cm)
                
            elif self.model_type == "UIRT":
                ability = self.workers[worker_id]
                difficulty, discrimination, guessing = self.tasks[task_id]
                prob = UIRT.compute_probability(ability, difficulty, discrimination, guessing)
                
            elif self.model_type == "CIRT":
                difficulty, discrimination, guessing, gt = self.tasks[task_id]
                ability = self.workers[worker_id][gt]
                prob = CIRT.compute_probability(ability, difficulty, discrimination, guessing)
            
        return prob
    
    def compute_log_likelihood(self, response_df):
        
        if self.model_type == "Dawid-Skene":
            llh = DS.compute_log_likelihood(response_df, self.workers)
        
        elif self.model_type == "UIRT":
            response_array, wid_dict, tid_dict = girth_format(response_df)
            
            abilities = np.zeros(len(wid_dict.keys()))
            num_tasks = len(tid_dict.keys())
            difficulties = np.zeros(num_tasks)
            discrimination = np.zeros(num_tasks)
            guessing = np.zeros(num_tasks)
            for wid, idx in wid_dict.items():
                ability = self.workers[wid]
                abilities[idx] = ability
            for tid, idx in tid_dict.items():
                diff, disc, guess = self.tasks[tid]
                difficulties[idx] = diff
                discrimination[idx] = disc
                guessing[idx] = guess
            llh = UIRT.compute_log_likelihood(response_array, abilities, difficulties, discrimination, guessing)
            
        elif self.model_type == "CIRT":
            df0, df1 = CIRT.get_dfs_by_gt(response_df)
            
            _, wid_dict0, tid_dict0 = girth_format(df0)
            _, wid_dict1, tid_dict1 = girth_format(df1)
            
            wid_dicts = {0: wid_dict0, 1: wid_dict1}
            tid_dicts = {0: tid_dict0, 1: tid_dict1}
            
            param_dict = {}
            for gt in [0, 1]:
                wid_dict = wid_dicts[gt]
                tid_dict = tid_dicts[gt]
                
                abilities = np.zeros(len(wid_dict.keys()))
                num_tasks = len(tid_dict.keys())
                difficulties = np.zeros(num_tasks)
                discrimination = np.zeros(num_tasks)
                guessing = np.zeros(num_tasks)
                
                for wid, idx in wid_dict.items():
                    ability = self.workers[wid][gt]
                    abilities[idx] = ability
                for tid, idx in tid_dict.items():
                    diff, disc, guess, _ = self.tasks[tid]
                    difficulties[idx] = diff
                    discrimination[idx] = disc
                    guessing[idx] = guess
                    
                param_dict[gt] = [wid_dict, tid_dict, abilities, difficulties, discrimination, guessing]
            
            llh = CIRT.compute_log_likelihood(response_df, param_dict)
        
        return llh
    
    def compute_aic_bic(self, response_df):
        
        if self.model_type == "Dawid-Skene":
            aic, bic = DS.compute_aic_bic(response_df, self.workers, "DS")
        
        elif self.model_type == "UIRT":
            response_array, wid_dict, tid_dict = girth_format(response_df)
            
            abilities = np.zeros(len(wid_dict.keys()))
            num_tasks = len(tid_dict.keys())
            difficulties = np.zeros(num_tasks)
            discrimination = np.zeros(num_tasks)
            guessing = np.zeros(num_tasks)
            for wid, idx in wid_dict.items():
                ability = self.workers[wid]
                abilities[idx] = ability
            for tid, idx in tid_dict.items():
                diff, disc, guess = self.tasks[tid]
                difficulties[idx] = diff
                discrimination[idx] = disc
                guessing[idx] = guess
            aic, bic = UIRT.compute_aic_bic(response_array, abilities, difficulties, discrimination, guessing, self.model_subtype)
            
        elif self.model_type == "CIRT":
            df0, df1 = CIRT.get_dfs_by_gt(response_df)
            
            _, wid_dict0, tid_dict0 = girth_format(df0)
            _, wid_dict1, tid_dict1 = girth_format(df1)
            
            wid_dicts = {0: wid_dict0, 1: wid_dict1}
            tid_dicts = {0: tid_dict0, 1: tid_dict1}
            
            param_dict = {}
            for gt in [0, 1]:
                wid_dict = wid_dicts[gt]
                tid_dict = tid_dicts[gt]
                
                abilities = np.zeros(len(wid_dict.keys()))
                num_tasks = len(tid_dict.keys())
                difficulties = np.zeros(num_tasks)
                discrimination = np.zeros(num_tasks)
                guessing = np.zeros(num_tasks)
                
                for wid, idx in wid_dict.items():
                    ability = self.workers[wid][gt]
                    abilities[idx] = ability
                for tid, idx in tid_dict.items():
                    diff, disc, guess, _ = self.tasks[tid]
                    difficulties[idx] = diff
                    discrimination[idx] = disc
                    guessing[idx] = guess
                    
                param_dict[gt] = [wid_dict, tid_dict, abilities, difficulties, discrimination, guessing]
            
            aic, bic = CIRT.compute_aic_bic(response_df, param_dict, self.model_subtype)
        
        return aic, bic