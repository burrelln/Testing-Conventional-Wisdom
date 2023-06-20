"""
Code for the experiments that are found in the paper and supplementary material.

@author: Noah Burrell <burrelln@umich.edu>

"""
import json
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import SeedSequence, default_rng
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, gaussian_kde
import seaborn as sns
from sklearn.mixture import GaussianMixture
from statistics import mean
"""
NOTE: The following package (modality, available at https://github.com/kjohnsson/modality) does not work out-of-the-box in Python 3. It can be modified to work with Python 3, mostly by adding parenthesis to print statements in the code. As a result of this issue, I have commented out the experimental code in this module that references the modality package. 
"""
# import modality

from models.model import Model
from models.cross_validation import kfold_cross_validation
import models.conditional_irt as CIRT


def get_df_bm():

    gt_filename="data/bm_gt.tsv"
    label_filename="data/bm_labels.tsv"

    gt_df=pd.read_csv(gt_filename, sep="\t", names=[
                      "task_id", "ground_truth"], index_col=0)
    labels_df=pd.read_csv(label_filename, sep="\t", names=[
                          "user_id", "task_id", "report"])

    df=labels_df.join(gt_df, on="task_id", how="inner")
    
    df.attrs['name'] = 'bm'
    df.attrs['id_num'] = 0
    df.attrs['model'] = 'C1PL'

    return df

def get_df_hcb():
    
    filename = "data/trec-rf10-data.tsv"

    df = pd.read_csv(filename, sep='\t', usecols=[
                     "workerID",	"docID",	 "gold",	 "label"])
    df = df.loc[(df["gold"].ge(0)) & (df["label"].ge(0))]
    df["ground_truth"] = df["gold"].gt(0).astype(int)
    df["report"] = df["label"].gt(0).astype(int)
    df = df.drop(["gold", "label"], axis=1)
    df = df.rename(columns={
            "workerID": "user_id",
            "docID": "task_id",
        }).reset_index(drop=True)
    
    df.attrs['name'] = 'hcb'
    df.attrs['id_num'] = 1
    df.attrs['model'] = 'DS'

    return df

def get_df_rte():
    
    filename = "data/rte.standardized.tsv"

    df = pd.read_csv(filename, sep="\t")
    df = df.drop("!amt_annotation_ids", axis=1)

    df = df.rename(columns={
            "!amt_worker_ids": "user_id",
            "orig_id": "task_id",
            "response": "report",
            "gold": "ground_truth"
        })
    
    df.attrs['name'] = 'rte'
    df.attrs['id_num'] = 2
    df.attrs['model'] = 'DS'

    return df

def get_df_temp():
    
    filename = "data/temp.standardized.tsv"

    df = pd.read_csv(filename, sep="\t")
    df = df.drop("!amt_annotation_ids", axis=1)

    df = df.rename(columns={
            "!amt_worker_ids": "user_id",
            "orig_id": "task_id",
            "response": "report",
            "gold": "ground_truth"
        })

    df["report"] = df["report"] - 1
    df["ground_truth"] = df["ground_truth"] - 1
    
    df.attrs['name'] = 'temp'
    df.attrs['id_num'] = 3
    df.attrs['model'] = 'DS'

    return df


def get_df_wb():
    
    gt_filename = "data/wb_gt.json"
    label_filename = "data/wb_labels.json"

    with open(gt_filename, "r") as gt_file:
        gt_dict = json.load(gt_file)

    with open(label_filename, "r") as label_file:
        label_dict = json.load(label_file)

    uids = []
    tids = []
    reports = []
    gts = []

    for uid, responses in label_dict.items():
        for task, response in responses.items():
            gt = int(gt_dict[task])
            uids.append(uid)
            tids.append(task)
            reports.append(int(response))
            gts.append(gt)

    df = pd.DataFrame({
                        "user_id": uids,
                        "task_id": tids,
                        "report": reports,
                        "ground_truth": gts
                    })
    
    df.attrs['name'] = 'wb'
    df.attrs['id_num'] = 4
    df.attrs['model'] = 'C1PL'

    return df

def get_df_wvscm():
    
    gt_filename = "data/wvscm_gt.tsv"
    label_filename = "data/wvscm_labels.txt"

    gt_df = pd.read_csv(gt_filename, sep="\t", names=[
                        "task_id", "ground_truth"], index_col=0)
    labels_df = pd.read_csv(label_filename, sep=" ", names=[
                            "task_id", "user_id", "report"])

    df = labels_df.join(gt_df, on="task_id", how="inner")
    
    df.attrs['name'] = 'wvscm'
    df.attrs['id_num'] = 5
    df.attrs['model'] = 'C1PL'

    return df

def get_df_sp():
    
    filename = "data/SP_amt.csv"

    df = pd.read_csv(filename, names=[
                     "user_id", "task_id", "report", "ground_truth", "time"])
    
    df.attrs['name'] = 'sp'
    df.attrs['id_num'] = 6
    df.attrs['model'] = 'C1PL'

    return df

def get_df_stats(df):
    
    task_df=df[["task_id", "ground_truth"]].groupby(["task_id"]).max()
    
    return task_df["ground_truth"].value_counts()


def cdn_permutation_test(df, median=False):
    
    if median:
        results=cdn_permutation_test_median(df)
    else:
        results=cdn_permutation_test_mean(df)
        
    return results
    
def cdn_permutation_test_mean(df):

    seed = 226824186500684295244439045377294328828
    seedseq = SeedSequence(seed)
    child_seeds = seedseq.spawn(7)
    
    id_num = df.attrs['id_num'] 
    rng = default_rng(child_seeds[id_num])
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    
    user_df = df.groupby(["user_id", "ground_truth"]).mean()
    avg_difference_actual = abs(user_df["correct"].xs(1, level="ground_truth") - user_df["correct"].xs(0, level="ground_truth")).mean()
    
    permutations = 999
    as_extreme = 0
    task_df = df[["task_id", "ground_truth"]].groupby(["task_id"]).max() 
        
    initial_num_columns = len(task_df.columns) 
    while len(task_df.columns) < permutations + initial_num_columns:
        i = len(task_df.columns) - 1
        key = f'permutation_{i}'
        task_df[key] = rng.permutation(task_df["ground_truth"])
        task_df = task_df.T.drop_duplicates().T
    
    for i in range(permutations):
        key = f'permutation_{i}'
        perm_df = df.join(task_df[key], on="task_id").rename(columns={key: "permutation"}) 
        
        user_df = perm_df.groupby(["user_id", "permutation"]).mean()
        avg_difference = abs(user_df["correct"].xs(1, level="permutation") - user_df["correct"].xs(0, level="permutation")).mean()
        
        if avg_difference >= avg_difference_actual:
            as_extreme += 1

    exact_p_val = (as_extreme + 1) / (permutations + 1)    

    return exact_p_val

def cdn_permutation_test_median(df):
    
    seed = 15085037779789102211404597529947168984
    seedseq = SeedSequence(seed)
    child_seeds = seedseq.spawn(7)
    
    id_num = df.attrs['id_num'] 
    rng = default_rng(child_seeds[id_num])
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    
    user_df = df.groupby(["user_id", "ground_truth"]).mean()
    median_difference_actual = abs(user_df["correct"].xs(1, level="ground_truth") - user_df["correct"].xs(0, level="ground_truth")).median()
    
    permutations = 999
    as_extreme = 0
    task_df = df[["task_id", "ground_truth"]].groupby(["task_id"]).max() 
    
    initial_num_columns = len(task_df.columns) 
    while len(task_df.columns) < permutations + initial_num_columns:
        i = len(task_df.columns) - 1
        key = f'permutation_{i}'
        task_df[key] = rng.permutation(task_df["ground_truth"])
        task_df = task_df.T.drop_duplicates().T
    
    medians = []
    for i in range(permutations):
        key = f'permutation_{i}'
        perm_df = df.join(task_df[key], on="task_id").rename(columns={key: "permutation"})
        
        user_df = perm_df.groupby(["user_id", "permutation"]).mean()
        median_difference = abs(user_df["correct"].xs(1, level="permutation") - user_df["correct"].xs(0, level="permutation")).median()
        
        medians.append(median_difference)
        
        if median_difference >= median_difference_actual:
            as_extreme += 1

    exact_p_val = (as_extreme + 1) / (permutations + 1)    

    return exact_p_val, (np.round(np.mean(medians), 3), np.round(np.median(medians), 3), np.round(np.max(medians), 3))

def category_dependence_confidence_interval(df):
    
    seed = 130967690422459519973625322108862826111
    seedseq = SeedSequence(seed)
    child_seeds = seedseq.spawn(7)
    
    id_num = df.attrs['id_num'] 
    rng = default_rng(child_seeds[id_num])
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    
    user_df = df.groupby(["user_id", "ground_truth"]).mean()
    median_difference_actual = abs(user_df["correct"].xs(1, level="ground_truth") - user_df["correct"].xs(0, level="ground_truth")).median()
    
    iterations = 1000
    values = np.zeros(iterations)
    
    for i in range(iterations):
        # Resample workers, then tasks from those workers
        resample_list = []
        re_user_df = user_df.sample(n=len(user_df.index), replace=True, random_state=rng).index.get_level_values(0)
        for new_id, old_id in enumerate(re_user_df):
            task_df = df.loc[df["user_id"].eq(old_id)]
            re_task_df = task_df.sample(n = len(task_df), replace=True, random_state=rng)
            re_task_df["user_id"] = np.full(len(re_task_df), new_id)
            resample_list.append(re_task_df)
            
        resampled_df = pd.concat(resample_list, ignore_index=True)
        
        new_user_df = resampled_df.groupby(["user_id", "ground_truth"]).mean()
        median_difference = abs(new_user_df["correct"].xs(1, level="ground_truth") - new_user_df["correct"].xs(0, level="ground_truth")).median()
        
        values[i] = median_difference
        
    sorted_vals = np.sort(values)
    lower = sorted_vals[5] 
    upper = sorted_vals[-6]       
    
    ci = (np.round(lower, 3), np.round(upper, 3))
    estimate = median_difference_actual

    return estimate, ci

def fit_test(df, _10FL, BIC):

    models=["DS", "1PL", "C1PL", "2PL", "C2PL", "3PL", "C3PL"]
    
    print(df.attrs['name'])

    if _10FL:
        size=int(round(len(df.index)/10))
        seed = 196524889463827959632304002031809580230
        seedseq = SeedSequence(seed)
        child_seeds = seedseq.spawn(7)
        id_num = df.attrs['id_num'] 
        
        s0=kfold_cross_validation(size, df, 0, -1, rng=default_rng(child_seeds[id_num]))
        s1=kfold_cross_validation(size, df, 1, 1, rng=default_rng(child_seeds[id_num]))
        s2=kfold_cross_validation(size, df, 3, 1, rng=default_rng(child_seeds[id_num]))
        s3=kfold_cross_validation(size, df, 1, 2, rng=default_rng(child_seeds[id_num]))
        s4=kfold_cross_validation(size, df, 3, 2, rng=default_rng(child_seeds[id_num]))
        scores=[s0, s1, s2, s3, s4]

        argmax=np.argmax(scores)

        model=models[argmax]

        print("KFL:", model)

    if BIC:
        m0=Model(0, -1)
        m1=Model(1, 1)
        m2=Model(3, 1)
        m3=Model(1, 2)
        m4=Model(3, 2)

        m0.fit(df)
        m1.fit(df)
        m2.fit(df)
        m3.fit(df)
        m4.fit(df)

        a0, b0=m0.compute_aic_bic(df)
        a1, b1=m1.compute_aic_bic(df)
        a2, b2=m2.compute_aic_bic(df)
        a3, b3=m3.compute_aic_bic(df)
        a4, b4=m4.compute_aic_bic(df)

        # Uncomment to also print out AIC score
        # a_scores=[a0, a1, a2, a3, a4]
        # argmin_a=np.argmin(a_scores)
        # model_a=models[argmin_a]
        # print("AIC:", model_a)

        b_scores=[b0, b1, b2, b3, b4]
        argmin_b=np.argmin(b_scores)
        model_b=models[argmin_b]
        print("BIC:", model_b)

def fast_fit_test(df, _10FL, BIC):

    models=["DS", "1PL", "C1PL", "2PL", "C2PL", "3PL", "C3PL"]
    
    print(df.attrs['name'])

    if _10FL:
        size=int(round(len(df.index)/10))
        seed = 196524889463827959632304002031809580230
        seedseq = SeedSequence(seed)
        child_seeds = seedseq.spawn(7)
        id_num = df.attrs['id_num'] 
        s0=kfold_cross_validation(size, df, 0, -1, rng=default_rng(child_seeds[id_num]))
        s1=kfold_cross_validation(size, df, 1, 1, rng=default_rng(child_seeds[id_num]))
        s2=kfold_cross_validation(size, df, 3, 1, rng=default_rng(child_seeds[id_num]))

        scores=[s0, s1, s2]

        argmax=np.argmax(scores)

        model=models[argmax]

        print("KFL:", model)

    if BIC:
        m0=Model(0, -1)
        m1=Model(1, 1)
        m2=Model(3, 1)

        m0.fit(df)
        m1.fit(df)
        m2.fit(df)

        a0, b0=m0.compute_aic_bic(df)
        a1, b1=m1.compute_aic_bic(df)
        a2, b2=m2.compute_aic_bic(df)

        # Uncomment to also print out AIC score
        # a_scores=[a0, a1, a2]
        # argmin_a=np.argmin(a_scores)
        # model_a=models[argmin_a]
        # print("AIC:", model_a)

        b_scores=[b0, b1, b2]
        argmin_b=np.argmin(b_scores)
        model_b=models[argmin_b]
        print("BIC:", model_b)

def estimate_GMM_classes(df, num_components):
    
    # NOTE: GaussianMixture does not accept an np.random.Generator object as a valid argument for the random_state parameter
    # It uses a legacy generator, via np.random.RandomState (or an integer seed)
    
    seeds = [
        # Seeds must be between 0 and 2**32 - 1
        3011443081,
        3145598136,
        2817830871,
        2923185601,
        712654709,
        75402559,
        4052459031
    ]

    id_num = df.attrs['id_num'] 
    rand_state = np.random.RandomState(seeds[id_num])
    
    wids, logit_probs=compute_logit_prob_correct(df)

    lps=np.array(logit_probs).reshape(-1, 1)
    gmm=GaussianMixture(n_components=num_components,
                        random_state=rand_state)
    assignments=gmm.fit_predict(lps)
    means=gmm.means_.T[0]

    if num_components == 2:
        novice=3
        worker=np.argmin(means)
        expert=np.argmax(means)

    elif num_components == 3:
        novice=np.argmin(means)
        expert=np.argmax(means)
        worker=list(set([0, 1, 2]) - set([novice, expert]))[0]

    labels=[]
    label_map={worker: "Amateur", expert: "Expert", novice: "Novice"}
    for label in assignments:
        labels.append(label_map[label])

    return wids, labels, logit_probs

def compute_logit_prob_correct_by_category(df, rng=None):
    
    model = df.attrs['model']

    wids=[]
    logit_probs=[]

    if model == "DS":

        m=Model(0, -1)
        m.fit(df)

        for wid, cm in m.workers.items():
            p=cm[0, 0]
            q=cm[1, 1]

            wids.append(wid)
            logit_probs.append((logit(p), logit(q)))

    elif model == "C1PL":

        m=Model(3, 1)
        m.fit(df)

        difficulty_lists=[[], []]
        discrimination_lists=[[], []]

        for tid, (diff, disc, guess, gt) in m.tasks.items():
            difficulty_lists[gt].append(diff)
            discrimination_lists[gt].append(disc)

        array0=np.array(difficulty_lists[0])
        array1=np.array(difficulty_lists[1])
        
        if rng is None:
            seed = 1418430346944640380543267523358816546
            seedseq = SeedSequence(seed)
            child_seeds = seedseq.spawn(7)
            id_num = df.attrs['id_num']
            rng = default_rng(child_seeds[id_num])

        kde0=gaussian_kde(array0, bw_method='silverman')
        kde1=gaussian_kde(array1, bw_method='silverman')

        num_samples=500

        points0=np.vstack((kde0.resample(num_samples, seed=rng), np.full(
            num_samples, discrimination_lists[0][0])))
        points1=np.vstack((kde1.resample(num_samples, seed=rng), np.full(
            num_samples, discrimination_lists[1][0])))

        for wid, d in m.workers.items():

            probs0 = []
            probs1 = []

            for diff, disc in points0.T:
                probs0.append(CIRT.compute_probability(d[0], diff, disc, 0))

            for diff, disc in points1.T:
                probs1.append(CIRT.compute_probability(d[1], diff, disc, 0))

            prob_correct0 = mean(probs0)
            prob_correct1 = mean(probs1)

            wids.append(wid)
            logit_probs.append((logit(prob_correct0), logit(prob_correct1)))

    else:
        print("Un-recognized model type.")
        return

    return wids, logit_probs, m

def compute_logit_prob_correct(df, return_m=False, rng=None):
    
    stats=get_df_stats(df)
    num0=stats[0]
    num1=stats[1]
    total=num0 + num1
    frac0=num0/total
    frac1=1.0 - frac0
    
    wids, split_lps, m = compute_logit_prob_correct_by_category(df, rng)

    model = df.attrs['model']

    if model == "DS":
        combine = lambda p, q: frac0*p + frac1*q
    elif model == "C1PL":
        combine = lambda p, q: mean([p, q])
    else:
        print("Un-recognized model type.")
        return
    
    logit_probs = []
    for lp, lq in split_lps:
        p = expit(lp)
        q = expit(lq)
        prob_correct = combine(p, q)
        lp=logit(prob_correct)
        logit_probs.append(lp)

    if return_m:
        return wids, logit_probs, m
    else:
        return wids, logit_probs

def compute_logit_prob_correct_no_outliers(df):
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    user_df = df.groupby(["user_id"]).correct.agg(["mean", "count"])
    expert_df = user_df.loc[user_df["mean"].eq(1.0)]
    spammer_df = user_df.loc[user_df["mean"].eq(0.0)]
    
    wids, logit_probs = compute_logit_prob_correct(df)
    
    wids_all = np.array(wids)
    lps_all = np.array(logit_probs)
    if len(spammer_df.index) > 0 and len(expert_df.index) > 0:
        wid_array = wids_all[(lps_all > lps_all.min()) & (lps_all < lps_all.max())]
        lps = lps_all[(lps_all > lps_all.min()) & (lps_all < lps_all.max())]
    elif len(expert_df.index) > 0:
        wid_array = wids_all[lps_all < lps_all.max()]
        lps = lps_all[lps_all < lps_all.max()]
    else:
        wid_array = wids_all
        lps = lps_all
        
    return list(wid_array), list(lps)
    
def test_expert_significance(df):
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    user_df = df.groupby(["user_id"]).correct.agg(["mean", "count"])
    expert_df = user_df.loc[user_df["mean"].eq(1.0)]
    spammer_df = user_df.loc[user_df["mean"].eq(0.0)]
    
    true_num_experts = len(expert_df.index)
    true_expert_total = expert_df["count"].sum()
    true_expert_avg_answered = true_expert_total / true_num_experts
    
    _, logit_probs=compute_logit_prob_correct(df)
    
    lps_all = np.array(logit_probs)
    if len(spammer_df.index) > 0:
        lps = lps_all[(lps_all > lps_all.min()) & (lps_all < lps_all.max())]
    else:
        lps = lps_all[lps_all < lps_all.max()]
    
    mean, std = norm.fit(lps)

    seed = 240794045991795426088744662229109482170
    seedseq = SeedSequence(seed)
    child_seeds = seedseq.spawn(7)
    id_num = df.attrs['id_num']
    rng = default_rng(child_seeds[id_num])
    
    num_iterations = 999
    as_extreme = 1
    for num in range(num_iterations):
        num_experts = 0
        expert_total = 0
        
        for count in user_df["count"]:
            lp = norm.rvs(mean, std, size=1, random_state=rng)
            p = expit(lp)
            num_correct = rng.binomial(count, p)
            if num_correct == count:
                num_experts += 1
                expert_total += num_correct
                
        expert_avg_answered = expert_total / num_experts
        
        # # Uncomment the next two lines and comment out the two lines after the next comment to get p_values for the total number of "experts"
        # if num_experts >= true_num_experts:
        #     as_extreme += 1
        # # Uncomment the next two lines and comment out the two previous lines to get p_values for avg number of questions answered by "experts"
        if expert_avg_answered >= true_expert_avg_answered:
            as_extreme += 1
            
    p_val = as_extreme / (num_iterations + 1)
            
    return p_val

def test_spammer_significance(df):
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    user_df = df.groupby(["user_id"]).correct.agg(["mean", "count"])
    spammer_df = user_df.loc[user_df["mean"].eq(0.0)]
    
    true_num_spammers = len(spammer_df.index)
    true_spammer_total = spammer_df["count"].sum()
    true_spammer_avg_answered = true_spammer_total / true_num_spammers
    
    _, logit_probs=compute_logit_prob_correct_no_outliers(df)
    
    lps_all = np.array(logit_probs)
    if len(spammer_df.index) > 0:
        lps = lps_all[(lps_all > lps_all.min()) & (lps_all < lps_all.max())]
    else:
        lps = lps_all[lps_all < lps_all.max()]
    
    mean, std = norm.fit(lps)
    
    seed = 188378128130564168455355667418867548996
    seedseq = SeedSequence(seed)
    child_seeds = seedseq.spawn(7)
    id_num = df.attrs['id_num']
    rng = default_rng(child_seeds[id_num])
    
    num_iterations = 999
    as_extreme = 1
    for num in range(num_iterations):
        num_spammers = 0
        spammer_total = 0
        
        for count in user_df["count"]:
            lp = norm.rvs(mean, std, size=1, random_state=rng)
            p = expit(lp)
            num_correct = rng.binomial(count, p)
            if num_correct == 0:
                num_spammers += 1
                spammer_total += count
                
        spammer_avg_answered = spammer_total / num_spammers
        
        # # Uncomment the next two lines and comment out the two lines after the next comment to get p_values for the total number of "spammers"
        # if num_spammers >= true_num_spammers:
        #    as_extreme += 1
        # # Uncomment the next two lines and comment out the two previous lines to get p_values for avg number of questions answered by "spammers"
        if spammer_avg_answered >= true_spammer_avg_answered:
             as_extreme += 1
            
    p_val = as_extreme / (num_iterations + 1)
            
    return p_val

"""
# The following function is commented out, because it references the modality package.

def test_logit_prob_correct_modality(df, diptest=True):

    _, logit_probs=compute_logit_prob_correct(df)

    lp_array=np.array(logit_probs)

    if diptest:
        p_val=modality.calibrated_diptest(
            lp_array, 0.05, 'shoulder', adaptive_resampling=False)
    else:
        p_val=modality.calibrated_bwtest(
            lp_array, 0.05, 'shoulder', I=(-20, 20), adaptive_resampling=False)

    if diptest:
        print(f"{df.attrs['name']} diptest: {p_val}")
    else:
        print(f"{df.attrs['name']} bwtest: {p_val}")

    return p_val
"""

"""
def test_logit_prob_correct_modality_no_outliers(df, diptest=True):
# The following function is commented out, because it references the modality package.

    _, logit_probs=compute_logit_prob_correct_no_outliers(df)

    lp_array=np.array(logit_probs)

    if diptest:
        p_val=modality.calibrated_diptest(
            lp_array, 0.05, 'shoulder', adaptive_resampling=False)
    else:
        p_val=modality.calibrated_bwtest(
            lp_array, 0.05, 'shoulder', I=(-20, 20), adaptive_resampling=False)

    if diptest:
        print(f"{df.attrs['name']} diptest: {p_val}")
    else:
        print(f"{df.attrs['name']} bwtest: {p_val}")

    return p_val
"""

def test_worker_homogeneity_using_irt(df):
    
    seed = 259236265427323999925729521915801063510
    seedseq = SeedSequence(seed)
    child_seeds = seedseq.spawn(7)
    
    wids, logit_probs, m = compute_logit_prob_correct(df, return_m=True)
    m.workers["average"] = m.ability_means.copy()
    
    id_num = df.attrs['id_num'] 
    rng = default_rng(child_seeds[id_num])
    
    real_test_stat = np.var(logit_probs)
    
    repetitions = 999
    as_extreme = 0
    test_stats = []
    
    df_copy = df[["user_id", "task_id", "ground_truth"]].copy().reset_index()
    initial_num_columns = len(df_copy.columns)
    
    while len(df_copy.columns) < repetitions + initial_num_columns:
        report_columns = None
        for idx, row in df_copy.iterrows():
            tid, gt = row["task_id"], row["ground_truth"]
            prob_correct = m.predict("average", tid)
            p = gt*prob_correct + (1-gt)*(1-prob_correct)
            num_samples = repetitions + initial_num_columns - len(df_copy.columns)
            responses = rng.binomial(1, p, size=num_samples)
            if report_columns is None:
                report_columns = responses
            else:
                report_columns = np.vstack((report_columns, responses))

        new_columns = pd.DataFrame({
                f'repetition_{i}': report_columns[:,i] for i in range(report_columns.shape[1])
            })
            
        df_copy = pd.concat([df_copy, new_columns], axis=1)
        df_copy = df_copy.T.drop_duplicates().T
        
    grandchildren = child_seeds[id_num].spawn(repetitions)
    for i in range(repetitions):
        key = f'repetition_{i}'
        new_df = df_copy[["user_id", "task_id", "ground_truth", key]].rename(columns={key: "report"}).reset_index()
        new_df.attrs = df.attrs.copy()
        new_rng = default_rng(grandchildren[i])
        _, new_logit_probs = compute_logit_prob_correct(new_df, rng=new_rng)
        test_stat = np.var(new_logit_probs)
        test_stats.append(test_stat)

    exact_p_val = (as_extreme + 1) / (repetitions + 1)

    return real_test_stat, exact_p_val, (np.mean(test_stats), np.median(test_stats), np.max(test_stats))

def investigate_diabolical_tasks(df, GMM_num_components):

    num_tasks=df['task_id'].nunique()

    wids, labels, _ = estimate_GMM_classes(df, GMM_num_components)

    label_df=pd.DataFrame(
        {'user_id': wids, 'label': labels}).set_index('user_id')

    df["correct"]=df["ground_truth"].eq(df["report"]).astype(int)

    df=df.join(label_df, how='inner', on='user_id')

    amateur_df=df.loc[df['label'].ne('Expert')][['task_id', 'correct']].groupby([
                                     'task_id']).correct.agg(['mean', 'count'])
    expert_df=df.loc[df['label'].eq('Expert')][['task_id', 'correct']].groupby([
                                    'task_id']).correct.agg(['mean', 'count'])

    user_df=amateur_df.join(expert_df, how='inner',
                            lsuffix='_am', rsuffix='_exp')

    hard_df = user_df.loc[(user_df['mean_am'].lt(0.5)) & (user_df['count_exp'].gt(1)) & (user_df['mean_exp'].gt(0.5))]

    frac_hard=len(hard_df.index) / num_tasks

    return frac_hard, hard_df

def permutation_test_tasks(df):

    seed = 208323165415474236086899782528375066379
    seedseq = SeedSequence(seed)
    child_seeds = seedseq.spawn(7)
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    
    df0 = df.loc[df["ground_truth"].eq(0)]
    df1 = df.loc[df["ground_truth"].eq(1)]
    
    task_df = df0.groupby("task_id").mean()
    sorted_array = task_df["correct"].sort_values().to_numpy()
    lower, upper = np.array_split(sorted_array, 2)
    real_test_stat0 = np.mean(upper) - np.mean(lower)
    
    task_df = df1.groupby("task_id").mean()
    sorted_array = task_df["correct"].sort_values().to_numpy()
    lower, upper = np.array_split(sorted_array, 2)
    real_test_stat1 = np.mean(upper) - np.mean(lower)
    
    real_test_stat = (real_test_stat0, real_test_stat1)
    
    permutations = 999
    as_extreme = [0, 0]
    
    id_num = df.attrs['id_num']
    rng = default_rng(child_seeds[id_num])

    test_stats = [[], []]
    for idx, cdf in enumerate((df0, df1)):
        cdf_copy = cdf[["user_id", "task_id", "correct"]].copy()
        cdf_copy = cdf_copy.T.drop_duplicates().T
        initial_num_columns = len(cdf_copy.columns) 
        while len(cdf_copy.columns) < permutations + initial_num_columns:
            i = len(cdf_copy.columns) - initial_num_columns
            key = f'permutation_{i}'
            cdf_copy[key] = rng.permutation(cdf["task_id"]) 
            cdf_copy = cdf_copy.T.drop_duplicates().T
            
        for i in range(permutations):
            key = f'permutation_{i}'
            perm_df = cdf_copy[[key, "correct"]].groupby(key).mean()
            sorted_array = perm_df["correct"].sort_values().to_numpy()
            lower, upper = np.array_split(sorted_array, 2)
            test_stat = np.mean(upper) - np.mean(lower)
            
            test_stats[idx].append(test_stat)
        
            if test_stat >= real_test_stat[idx]:
                as_extreme[idx] += 1

    exact_p_vals = ((as_extreme[0] + 1) / (permutations + 1), (as_extreme[1] + 1) / (permutations + 1))

    return real_test_stat, exact_p_vals, ((np.mean(test_stats[0]),np.mean(test_stats[1])), (np.median(test_stats[0]), np.median(test_stats[1])), (np.max(test_stats[0]), np.max(test_stats[1])))

def permutation_test_tasks_median(df):
    
    seed = 246397586945626455542589929530066130710
    seedseq = SeedSequence(seed)
    child_seeds = seedseq.spawn(7)
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    
    df0 = df.loc[df["ground_truth"].eq(0)]
    df1 = df.loc[df["ground_truth"].eq(1)]
    
    task_df = df0.groupby("task_id").mean()
    sorted_series = task_df["correct"].sort_values()
    real_test_stat0 = sorted_series.quantile(0.75) - sorted_series.quantile(0.25) 
    
    task_df = df1.groupby("task_id").mean()
    sorted_series = task_df["correct"].sort_values()
    real_test_stat1 = sorted_series.quantile(0.75) - sorted_series.quantile(0.25) 
    
    real_test_stat = (real_test_stat0, real_test_stat1)
    
    permutations = 999
    as_extreme = [0, 0]
    
    id_num = df.attrs['id_num']
    rng = default_rng(child_seeds[id_num])

    test_stats = [[], []]
    for idx, cdf in enumerate((df0, df1)):
        cdf_copy = cdf[["user_id", "task_id", "correct"]].copy()
        initial_num_columns = len(cdf_copy.columns) 
        while len(cdf_copy.columns) < permutations + initial_num_columns:
            i = len(cdf_copy.columns) - initial_num_columns
            key = f'permutation_{i}'
            cdf_copy[key] = rng.permutation(cdf["task_id"]) 
            cdf_copy = cdf_copy.T.drop_duplicates().T
            
        for i in range(permutations):
            key = f'permutation_{i}'
            perm_df = cdf_copy[[key, "correct"]].groupby(key).mean()
            sorted_series = perm_df["correct"].sort_values()
            test_stat = sorted_series.quantile(0.75) - sorted_series.quantile(0.25) 
            
            test_stats[idx].append(test_stat)
        
            if test_stat >= real_test_stat[idx]:
                as_extreme[idx] += 1

    exact_p_vals = ((as_extreme[0] + 1) / (permutations + 1), (as_extreme[1] + 1) / (permutations + 1))

    return real_test_stat, exact_p_vals, ((np.mean(test_stats[0]),np.mean(test_stats[1])), (np.median(test_stats[0]), np.median(test_stats[1])), (np.max(test_stats[0]), np.max(test_stats[1])))

def permutation_test_workers(df):
    
    seed = 216644054828115066273030857946391621791
    seedseq = SeedSequence(seed)
    child_seeds = seedseq.spawn(7)
    
    df["correct"] = df["ground_truth"].eq(df["report"]).astype(int)
    
    df0 = df.loc[df["ground_truth"].eq(0)]
    df1 = df.loc[df["ground_truth"].eq(1)]
    
    user_df0 = df0.groupby("user_id").mean()
    sorted_array = user_df0["correct"].sort_values().to_numpy()
    lower, upper = np.array_split(sorted_array, 2)
    real_test_stat0 = np.mean(upper) - np.mean(lower)
    
    user_df1 = df1.groupby("user_id").mean()
    sorted_array = user_df1["correct"].sort_values().to_numpy()
    lower, upper = np.array_split(sorted_array, 2)
    real_test_stat1 = np.mean(upper) - np.mean(lower)
    
    real_test_stat = (real_test_stat0, real_test_stat1)
    
    permutations = 999
    as_extreme = [0, 0]
    
    id_num = df.attrs['id_num']
    rng = default_rng(child_seeds[id_num])

    test_stats = [[], []]
    for idx, cdf in enumerate((df0, df1)):
        cdf_copy = cdf[["user_id", "task_id", "correct"]].copy()
        initial_num_columns = len(cdf_copy.columns) 
        while len(cdf_copy.columns) < permutations + initial_num_columns:
            i = len(cdf_copy.columns) - initial_num_columns
            key = f'permutation_{i}'
            cdf_copy[key] = rng.permutation(cdf["user_id"]) 
            cdf_copy = cdf_copy.T.drop_duplicates().T
            
        for i in range(permutations):
            key = f'permutation_{i}'
            perm_df = cdf_copy[[key, "correct"]].groupby(key).mean()
            sorted_array = perm_df["correct"].sort_values().to_numpy()
            lower, upper = np.array_split(sorted_array, 2)
            test_stat = np.mean(upper) - np.mean(lower)
            
            test_stats[idx].append(test_stat)
        
            if test_stat >= real_test_stat[idx]:
                as_extreme[idx] += 1

    exact_p_vals = ((as_extreme[0] + 1) / (permutations + 1), (as_extreme[1] + 1) / (permutations + 1))

    return real_test_stat, exact_p_vals, ((np.mean(test_stats[0]),np.mean(test_stats[1])), (np.median(test_stats[0]), np.median(test_stats[1])), (np.max(test_stats[0]), np.max(test_stats[1])))

def plot_df_logit_prob_correct_cirt():

    bmdf = get_df_bm()
    wbdf=get_df_wb()
    wvscmdf=get_df_wvscm()
    spdf=get_df_sp()

    ax=plt.gca()

    ylab=r'Density'
    xlab=r'Logit-Probability of Correctness'

    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)

    title=r'Category-dependent 1PL Estimates'

    for i, df in [(0, bmdf), (1, spdf), (2, wbdf), (3, wvscmdf)]:
        
        name = df.attrs['name']
        plt_name = name.upper()
        
        _, logit_probs = compute_logit_prob_correct(df)
        _ = sns.kdeplot(x=logit_probs, bw_method='silverman', ax=ax,
                      color=sns.color_palette()[i], label=plt_name)
        
    ax.set_xlim(-3, 5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4, 5]) 
    
    ax.set_ylim(0.00, 2.1)
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])

    ax.legend()

    plt.title(title)
    plt.tight_layout()
    figure_file="figures/kde-logit-prob_cirt.pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()
    
def plot_df_logit_prob_correct_ds():
    
    hcbdf=get_df_hcb()
    tempdf=get_df_temp()
    rtedf=get_df_rte()

    ax=plt.gca()

    ylab=r'Density'
    xlab=r'Logit-Probability of Correctness'

    ax.set_ylabel(ylab) 
    ax.set_xlabel(xlab)

    title=r'Dawid-Skene Estimates'

    for i, df in [(-1, hcbdf), (-2, rtedf), (-3, tempdf)]:
        
        name = df.attrs['name']
        plt_name = name.upper()
        
        _, logit_probs = compute_logit_prob_correct_no_outliers(df) # Note: Ignores outliers who respond correctly to all of their tasks (In the case of HCB, also ignores outliers who respond incorrectly to all of their tasks)
        _ = sns.kdeplot(x=logit_probs, bw_method='silverman', ax=ax,
                      color=sns.color_palette()[i], label=plt_name)
        
    ax.set_xlim(-3, 5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4, 5]) 
    
    ax.set_ylim(0.00, 2.1)
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])

    ax.legend()

    plt.title(title)
    plt.tight_layout()
    figure_file="figures/kde-logit-prob_ds.pdf"
    plt.savefig(figure_file, dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":

    df_list = [
        # This list loads DataFrames for each the data sets that we reference in the paper
        # Comment out any of the following lines to exclude the corresponding data set from an experiment
        get_df_bm(),
        get_df_hcb(),
        get_df_rte(),
        get_df_temp(),
        get_df_wb(),
        get_df_wvscm(),
        get_df_sp()
    ]
    
    """
    
    GUIDE TO RUNNING EXPERIMENTS
    ============================
    
    To perform an experiment, uncomment the block of code corresponding to that experiment in the `for` loop below and run this script.
    For experiments that correspond to a table in the paper or supplementary material, the experiment is labeled with the caption from the corresponding table in the first print statement of the code block (beneath the `if i == 0:`).
    
    I recommend running experiments one at a time, since they can take a long time to run. 
    (Also, the print statements that output the results of each experiment are written under the assumption that experiments will be run in this way.)
    
    """
    
    for i, df in enumerate(df_list):
        
        name = df.attrs["name"]
        
        # if i == 0:
        #     print("Summary of Each Data Set")
        # worker_df = df.groupby("user_id").count()
        # task_df = df.groupby("task_id").count()
        # category_df = df.groupby("ground_truth").count()
        # stats = f'{name} stats. workers: {len(worker_df.index)}; tasks: {len(task_df.index)}; responses (gt = 0): {category_df.iloc[0]["task_id"]}; responses (gt = 1): {category_df.iloc[1]["task_id"]}.'
        # print(stats)
        
        
        # if i == 0:
        #     print("Constructing 95% Confidence Intervals: Testing Null Hypothesis of Category Independence")
        # est, ci = category_dependence_confidence_interval(df)
        # result = f'{name} results. Estimate: {est}; CI: {ci}'
        # print(result)
        
        
        # if i == 0:
        #     print("Randomization Inference: Testing Null Hypothesis of Category Independence")
        # p_val, tup = cdn_permutation_test(df, median=True)
        # result = f'{name} results. pval: {p_val}; Avg/Med/Max: {tup}'
        # print(result)
        
        
        # if i == 0:
        #     print("Randomization Inference: Testing Null Hypothesis of Category Independence (Test Statistic = mean, mentioned in footnote)")
        # p_val = cdn_permutation_test(df, median=False)
        # result = f'{name} results. pval: {p_val}'
        # print(result)
        
        
        # if i == 0:
        #     print("Randomization Inference: Testing Null Hypothesis of Task Homogeneity")
        # test_stats, p_val, tup = permutation_test_tasks(df)
        # result = f'{name} results. observed: {test_stats}; pval: {p_val}; Avg/Med/Max: {tup}'
        # print(result)
        
        
        # if i == 0:
        #     print("Model Fitting: Best-Fitting Model for each Data Set")
        # if name in ['hcb']:
        #     fast_fit_test(df, _10FL=True, BIC=True)
        # else:
        #     fit_test(df, _10FL=True, BIC=True)
        
        
        # if i == 0:
        #     print("Randomization Inference: Testing Null Hypothesis of Worker Homogeneity")
        # test_stats, p_val, tup = permutation_test_workers(df)
        # result = f'{name} results. observed: {test_stats}; pval: {p_val}; Avg/Med/Max: {tup}'
        # print(result)
        
        
        # if i == 0:
        #     print("Examining Experts (DS Data Sets): Testing Expert (and, for HCB, Spammer) Significance (mentioned in text and footnote)")
        # if name in ['rte', 'temp']:
        #     result = test_expert_significance(df)
        #     print(f'{name} result. pval: {result}')
        # elif name in ['hcb']:
        #     result = test_expert_significance(df)
        #     print(f'{name} result (expert). pval: {result}')
        #     result = test_spammer_significance(df)
        #     print(f'{name} result (spammer). pval: {result}')
        
        """
        '''
        NOTE 1: These tests are commented out, because they involve the modality package.
        
        NOTE 2: The results of these tests are non-deterministic, because there is not currently a way to seed the tests implemented in the modality package.
        '''
        if i == 0:
            print("Modality Test: Testing Null Hypothesis of Unimodality")
        if name in ['bm', 'wb', 'wvscm', 'sp']:
            test_logit_prob_correct_modality(df, True)
            test_logit_prob_correct_modality(df, False)
        else:
            test_logit_prob_correct_modality_no_outliers(df, True)
            test_logit_prob_correct_modality_no_outliers(df, False)
        """
        
        # if i == 0:
        #     print("Model-Informed Resampling Test: Testing Null Hypothesis of Worker Homogeneity (C1PL Data Sets)")
        # if name not in ['bm', 'wb', 'wvscm', 'sp']:
        #     continue
        # test_stat, p_val, tup = test_worker_homogeneity_using_irt(df)
        # result = f'{name} results. observed: {test_stat}; pval: {p_val}; Avg/Med/Max: {tup}'
        # print(result)
        
        
        # if name == "wb":
        #     print("Further Exploring Task Heterogeneity: Diabolical Tasks (WB Data Set)")
        # if name not in ['wb']:
        #     continue
        # frac, hard_df = investigate_diabolical_tasks(df, 2)
        # hard_df["total"] = (hard_df["mean_am"]*hard_df["count_am"] + hard_df["mean_exp"]*hard_df["count_exp"]) / (hard_df["count_am"] + hard_df["count_exp"])
        # maj_right = hard_df.loc[hard_df["total"] > 0.5]
        # print(f'Total number of Diabolical Tasks: {len(hard_df.index)}.')
        # print(f'Total number of Diabolical Tasks where a majority of workers are correct: {len(maj_right.index)}.')
    
    """
    Uncomment the following two functions to create the two plots that appear in the paper (and save them as `.pdf` files, in the `figures` directory of the repo) 
    """
    # plot_df_logit_prob_correct_cirt()
    # plot_df_logit_prob_correct_ds()