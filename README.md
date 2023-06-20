# Testing-Conventional-Wisdom
This repository contains the code for the experiments in the forthcoming paper "Testing Conventional Wisdom (of the Crowd)" \[Burrell and Schoenebeck, 2023\], which will appear in the proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI 2023).

## Navigating the repository
A mini-guide:
- The file `experiments.py` contains the code for running each experiment in the paper; scroll down to the bottom of the file (after `if __name__ == "__main__"`) for more detailed instructions for running experiments.
- The `data` directory is empty in this repo, but is used to store the data sets that are used in our experiments, which can be downloaded from other sources according to the instructions below.
- The `figures` directory contains `.pdf` files of the plots that appear in the paper.
- The `models` directory contains the modules for fitting models to the data sets and evaluating the fit of each model. This includes an abstract Model class and modules for each specific model families that are called by that class.
- The `results` directory contains a `.txt` file with the experimental results (i.e., the output from running particular experiments) that are discussed in the paper and which can be replicated by running `experiments.py`. 
    
If you have questions or see what looks like a bug, let me know!

## Dependencies
The project is implemented in Python 3. The experiments rely on several well-known Python packages: `NumPy`, `pandas`, `Scikit-learn`, `SciPy`, and `seaborn`. All of these packages are available in the [Anaconda Python Distrbution](https://www.anaconda.com/products/individual).

The two additional dependences (not available from Anaconda) are:
  1. The [G. Item Response Theory](https://github.com/eribean/girth) (`girth`) package. 
  2. The [modality](https://github.com/kjohnsson/modality) package. 

NOTE: The `modality` package does not work out-of-the-box in Python 3. It can be modified to work with Python 3, mostly by adding parenthesis to print statements in the code. As a result of this issue, I have commented out the experimental code in this module that references the modality package. 
I may add the modified code to this repo at some point in the future; for now, I have commented out the code that relies on that package in `experiments.py`.

## Data Sets
To replicate the experiments from the paper, you will need to first download the relevant data set(s) according to the following instructions:

- BM data set available from: <https://github.com/ipeirotis/Get-Another-Label/tree/master/data/BarzanMozafari>
  To use this data set:
    1. Download the file `evaluation.txt` and rename it as `bm_gt.tsv`
    2. Download the file `labels.txt` and rename it as `bm_labels.tsv`
    3. Place both `.tsv` files in the `data` directory of this repository 
- HCB data set available from: <http://www.ischool.utexas.edu/~ml/data/trec-rf10-crowd.tgz>
  To use this data set:
    1. Download the file `trec-rf10-crowd.tgz` and unzip it
    2. Extract the file `trec-rf10-data.txt` and rename it as `trec-rf10-data.tsv`
    3. Place the `.tsv` files in the `data` directory of this repository 
- RTE and TEMP data sets are no longer available from their original source: <http://sites.google.com/site/nlpannotations/>
  But, they can be downloaded via the Wayback Machine from the Internet Archive [here](https://web.archive.org/web/20230331023329/http://sites.google.com/site/nlpannotations/) (for the original site) and [here](https://web.archive.org/web/20201021140217/https://756f1270-a-62cb3a1a-s-sites.googlegroups.com/site/nlpannotations/all_collected_data.tgz?attachauth=ANoY7crmLE0WcwWFsxwsy30TpjYxBjbh8C3CGWj9uFzurtqRJIBHNd4covkZDc6-bSaAAAjj9uPyK7THGCXiDMOyWi1g6_-5PRbwbLXQbnSps1OCsSKQVqGHNSCIT3kukSGqX06rIj5ogBhdjlVXzItZeTGknPviDk_4n5tpRz_dVKz_FBw6HlCWDfFeJs3lLlSdnXxLHOMkhhbs1S09Oq3X8jNe53rBEdmIIS5IWFuVI4zjbTO2rtQ%3D&attredirects=0) (to download the `.tgz` file).
  To use these data sets:
    1. Download the file `all_collected_data.tgz` and unzip it
    2. Extract the files `rte.standardized.tsv` and `temp.standardized.tsv`
    3. Place the `.tsv` files in the `data` directory of this repository 
- WB data set available from: <https://github.com/welinder/cubam/tree/public/demo/bluebirds>
  To use this data set:
    1. Download the file `gt.yaml` and convert it to a `.json` file named `wb_gt.json` (e.g., by using an online converter)
    2. Download the file `labels.yaml` and convert it to a `.json` file named `wb_labels.json` 
    3. Place the `.json` files in the `data` directory of this repository
- WVSCM data set available from: <https://inc.ucsd.edu/mplab/users/jake/DuchenneExperiment/DuchenneExperiment.html>
  To use this data set:
    1. Download the file `groundtruth.txt` and rename it as `wvscm_gt.tsv`
    2. Download the files `mturklabels.txt` and rename it as `wvscm_labels.txt` (note the different file extension here) 
    3. Place both the `.tsv` file and the `.txt` file in the `data` directory of this repository
- SP data set available from: <https://dx.doi.org/10.5258/SOTON/376544>
  To use this data set:
    1. Download the file `SP_amt.csv`
    2. Place both the `.csv` file in the `data` directory of this repository
    
To run our experiments using a different data set than those listed above:
1. Start by putting the relevant files in the `data` directory of this repository. 
2. Then, in `experiments.py` add a `get_df_{name}()` function that reads the data from the files and builds a pandas DataFrame of responses with the following columns:
     - `"user_id"`      : unique value that identifies an individual worker
     - `"task_id"`      : unique value that identifies an individual task
     - `"report"`       : response (0 or 1) provided by worker "user_id" on task "task_id"
     - `"ground_truth"` : correct response (0 or 1) for task "task_id" 
    Each row of the DataFrame should correspond to one response (i.e., the report of one worker on one task) in the data set.
    
    The DataFrame that is returned by the `get_df_{name}()` should be assigned the following attributes (in the `df.attrs` dictionary):
     - `"name"`  : a short name to identify the data set (ideally, the same one that is used in the name of the function)
     - `"id_num"`: the int 7, for the first new data set, then 8, then 9, and so on for each new data set that is added. (Note we start numbering new data sets with 7, because the data sets from the paper are numbered 0 through 6.)
     - `"model"` : a string with the name of the model that best fits the data according to a fit test (`'DS'`. `'C1PL'`, etc.). 
     (This can be added to the `get_df_{name}()` function after performing a fit test to find the best-fitting model. Note that if the best fitting model is not `'DS'` or `'C1PL'`, some modifications may need to be made to the experiments. They are only designed to handle those two cases, because those are the only cases in the data sets from the paper.)
3. For any experiments with randomness that you want to perform, update the number of seed sequences to be spawned from the initial seed to be equal to the number of data sets (7 plus however many new data sets you want to include) by changing the argument `x` of the `seedseq.spawn(x)` function. 
(For the `estimate_GMM_classes` function, which is called by the `investigate_diabolical_tasks` function, simply add a new seed to the list of seeds for each new data set.)
4. Under `if __name__ == "__main__"`, add the new function(s) `get_df_{name}()` to the end of the `df_list` that is constructed.
5. Uncomment the experiment that you would like to run and the results for your new data set will be included! 