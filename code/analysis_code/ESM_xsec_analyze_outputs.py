#!/usr/bin/env python

import os
import glob 
import sys
import shutil 
import re

import nibabel as nib

from argparse import ArgumentParser

import pandas as pd
import numpy as np
import nilearn.plotting as plotting
import itertools
import matplotlib.colors as colors
import seaborn as sns

import matplotlib.pyplot as plt
import math
from statannot import add_stat_annotation

sys.path.insert(0,'..')
import ESM_utils as esm

from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from nilearn import input_data, image

## look at mutation differences
def plot_aggregate_roi_performance(esm_output, output_dir, ref_pattern): 
    plt.figure(figsize=(5,5))
    sns.regplot(esm_output[ref_pattern].mean(1), esm_output['model_solutions0'].mean(1), color="indianred")
    r,p = stats.pearsonr(esm_output[ref_pattern].mean(1), esm_output['model_solutions0'].mean(1))
    r2 = r ** 2 
    xmin = np.min(esm_output[ref_pattern].mean(1))
    ymax = np.max(esm_output['model_solutions0'].mean(1))
    print(ymax)
    plt.text(xmin, ymax, "$r^2$ = {0}".format(np.round(r2, 2), fontsize=16))
    plt.xticks(x=16)
    plt.yticks(y=16)
    plt.xlabel(r"Observed A$\beta$ Probabilities", fontsize=16)
    plt.ylabel(r"Predicted A$\beta$ Probabilities", fontsize=16)
    plt.title(r"Average A$\beta$ Pattern Across All MC", fontsize=16)
    output_path = os.path.join(output_dir, "aggreggate_roi_performance_xsec.png")
    plt.tight_layout()
    plt.savefig(output_path)

def plot_hist_subject_performance(res, output_dir): 
    plt.figure(figsize=(5,5))
    g = sns.stripplot(x="mutation_type", y="model_r2", data=res, hue="AB_Positive", dodge=True)
    g.set(xticklabels=["PSEN1", "PSEN2", "APP"])
    plt.xlabel("Mutation Type")
    plt.ylabel("Within subject r2")
    plt.title("ESM X-Sec Performance Across Mutation Types")
    output_path = os.path.join(output_dir, "within_subject_xsec_perform_across_muttypes.png")
    plt.tight_layout()
    plt.savefig(output_path)

def set_ab_positive(ref_pattern_df, early_acc_rois):
    cols_to_analyze = [] 
    for col in ref_pattern_df.columns: 
        for roi in early_acc_rois: 
            if roi in col.lower(): 
                cols_to_analyze.append(col)
    for sub in ref_pattern_df.index:
        avg_ab_val = np.mean(list(ref_pattern_df.loc[sub, cols_to_analyze]))
        if avg_ab_val > 0.2: 
            ref_pattern_df.loc[sub, 'AB_Positive'] = True
        else: 
            ref_pattern_df.loc[sub, 'AB_Positive'] = False
    return ref_pattern_df

def get_mutation_type(subs, genetic_df):
    mutation_type = [] 
    for sub in subs: 
        mt = genetic_df[(genetic_df.IMAGID == sub)].MUTATIONTYPE.values[0]
        mutation_type.append(mt)
    return mutation_type

def get_eyo(subs, visit_labels, clinical_df):
    eyos = []
    for i, sub in enumerate(subs):
        eyo = clinical_df[(clinical_df.IMAGID == sub) & (clinical_df.visit == visit_labels[i])].DIAN_EYO.values[0]
        eyos.append(eyo)
    return eyos

def main(): 
    parser = ArgumentParser()
    parser.add_argument("filename",
                        help="Please pass base filename of ESM output file to analyze")
    parser.add_argument("--scale", 
                        type=bool, 
                        default=False, 
                        help="Whether or not the input had been sigmoid normalized prior to running ESM.")

    results = parser.parse_args()
    scale = results.scale

    if scale == True: 
        ref_pattern = "ref_pattern_orig"
    else: 
        ref_pattern = "ref_pattern"

    genetic_df = pd.read_csv("../../data/DIAN/participant_metadata/GENETIC_D1801.csv")
    clinical_df = pd.read_csv("../../data/DIAN/participant_metadata/CLINICAL_D1801.csv")

    esm_output_file = "../../data/DIAN/esm_output_mat_files/" + results.filename + ".mat"
    esm_output = esm.loadmat(esm_output_file)

    ref_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=esm_output['roi_labels'], 
                                 data=esm_output[ref_pattern].transpose())
    pred_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=esm_output['roi_labels'], 
                                 data=esm_output['model_solutions0'].transpose())
    early_acc_rois = ["precuneus", "medial orbitofrontal", "posterior cingulate", "caudate", "putamen"] 
    ref_pattern_df = set_ab_positive(ref_pattern_df, early_acc_rois)
    subs = esm_output['sub_ids']
    visit_labels = esm_output['visit_labels']
    roi_labels = esm_output['roi_labels']

    # make a new directory for figs corresponding to a specific output? 
    output_dir = os.path.join("../../figures", results.filename)
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir)
    
    plot_aggregate_roi_performance(esm_output, output_dir, ref_pattern)

    res = esm.Evaluate_ESM_Results(esm_output_file,
                                   sids=subs,
                                   labels=roi_labels,
                                   lit=True,
                                   plot=False)

    for i, sub in enumerate(res.index): 
        res.loc[sub, 'esm_idx'] = i 
    
    res['mutation_type'] = get_mutation_type(res.index, genetic_df)
    res['visit_label'] = visit_labels
    res['EYO'] = get_eyo(res.index, res.visit_label, clinical_df)
    res['AB_Positive'] = ref_pattern_df['AB_Positive']

    cols_to_evaluate = list(roi_labels)
    for roi in roi_labels:  
        if "thalamus" in roi.lower() or "globus pallidus" in roi.lower():
            cols_to_evaluate.remove(roi)
    print(len(cols_to_evaluate))
    r2_ab_positive = stats.pearsonr(ref_pattern_df.loc[:,cols_to_evaluate].mean(0), 
                                    pred_pattern_df.loc[:, cols_to_evaluate].mean(0))[0] ** 2 
    print(r2_ab_positive)                               
    
    plot_hist_subject_performance(res, output_dir)

if __name__ == "__main__":
    main()
    
