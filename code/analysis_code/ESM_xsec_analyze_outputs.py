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
import bct

sys.path.insert(0,'..')
import ESM_utils as esm

from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

## look at mutation differences

def plot_aggregate_roi_performance(ref_pattern_df, pred_pattern_df, roi_labels, output_dir): 
    plt.figure(figsize=(5,5))
    ref_roi_avg = np.mean(ref_pattern_df.loc[:, roi_labels], axis=0)
    pred_roi_avg = np.mean(pred_pattern_df.loc[:, roi_labels], axis=0)
    sns.regplot(ref_roi_avg, pred_roi_avg, color="indianred")
    r,p = stats.pearsonr(ref_roi_avg, pred_roi_avg)
    r2 = r ** 2 
    xmin = np.min(ref_roi_avg)
    ymax = np.max(pred_roi_avg)
    plt.text(xmin, ymax, "$r^2$ = {0}".format(np.round(r2, 2), fontsize=16))
    plt.xlim([-0.01,ymax+.1])
    plt.ylim([-0.01,ymax+.1])
    plt.xticks(x=16)
    plt.yticks(y=16)
    plt.xlabel(r"Observed A$\beta$ Probabilities", fontsize=16)
    plt.ylabel(r"Predicted A$\beta$ Probabilities", fontsize=16)
    plt.title(r"Average A$\beta$ Pattern Across All MC", fontsize=16)
    output_path = os.path.join(output_dir, "aggreggate_roi_performance_xsec.png")
    plt.tight_layout()
    plt.savefig(output_path)

def plot_aggregate_roi_performance_across_eyo(ref_pattern_df, pred_pattern_df, roi_labels, output_dir):  
    #threshold = 0.3
    age_ranges = [[-20, -10], [-10,0], [0,10], [10,20]] 
    n_cols = len(age_ranges)
    fig = plt.figure(figsize=(8,8))

    for idx, ar in enumerate(age_ranges):
        #loc = 2*(idx+1)-1
        ref = np.mean(ref_pattern_df[(ref_pattern_df.DIAN_EYO > ar[0]) & (ref_pattern_df.DIAN_EYO < ar[1])][roi_labels], axis=0)
        pred = np.mean(pred_pattern_df[(pred_pattern_df.DIAN_EYO > ar[0]) & (pred_pattern_df.DIAN_EYO < ar[1])][roi_labels], axis=0)
        n = ref_pattern_df[(ref_pattern_df.DIAN_EYO > ar[0]) & (ref_pattern_df.DIAN_EYO < ar[1])].shape[0]
        r,p = stats.pearsonr(ref, pred)
        r2 = r**2
        ax1 = plt.subplot(2, 2, idx+1)
        sns.regplot(ref, pred, color="indianred")
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.text(s=r"$r^2$ = {0}".format(np.round(r2, 2)), x=0, y=.9, fontsize=18)
        if idx < 2: 
            plt.xticks([])
        else:
            plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title("EYO: {0} to {1} (n = {2})".format(str(ar[0]), str(ar[1]), str(n)), fontsize=18)
    fig.text(0.2, 0, r"Observed A$\beta$ Probabilities", fontsize=26)
    fig.text(0.01, 0.2, r"Predicted A$\beta$ Probabilities", fontsize=26, rotation=90)
    #fig.text(0.2, .95, "Group-level ESM Performance", fontsize=26)
    output_path = os.path.join(output_dir, "aggreggate_roi_performance_xsec_across_eyo.png")
    #plt.tight_layout()
    plt.savefig(output_path)

def roi_performance_hist(ref_pattern_df, pred_pattern_df, roi_labels, output_dir):
    roi_r2 = []
    for roi in roi_labels:
        r = stats.pearsonr(ref_pattern_df.loc[:, roi], pred_pattern_df.loc[:, roi])[0]
        roi_r2.append(r**2)
    roi_r2_df = pd.DataFrame(columns=["ROI", "r2"])
    roi_r2_df['ROI'] = roi_labels 
    roi_r2_df['r2'] = roi_r2
    
    g = sns.catplot(x='ROI', y='r2',data=roi_r2_df, ci=None, 
                       order = roi_r2_df.sort_values('r2',ascending=False)['ROI'],
                       kind='bar')
    g.set_xticklabels(rotation=90)
    g.fig.set_size_inches((14,6))
    plt.title('R2 Per ROI Across All Subjects')
    output_path = os.path.join(output_dir, "roi_performance_hist.png")
    plt.tight_layout()
    plt.savefig(output_path)

def plot_subject_performance(res, output_dir, dataset): 
    plt.figure(figsize=(5,5))
    g = sns.boxplot(x="mutation_type", y="model_r2", data=res, hue="AB_Positive", dodge=True)
    sns.stripplot(x="mutation_type", y="model_r2", data=res, hue="AB_Positive", dodge=True, ax=g, color="black")
    g.set(xticklabels=["PSEN1", "PSEN2", "APP"])
    plt.xlabel("Mutation Type")
    plt.ylabel("Within subject r2")
    plt.title("ESM X-Sec Performance Across Mutation Types")
    output_path = os.path.join(output_dir, "within_subject_xsec_perform_across_muttypes.png")
    plt.tight_layout()
    plt.savefig(output_path)

def set_ab_positive(ref_pattern_df, rois_to_analyze):
    for sub in ref_pattern_df.index:
        avg_ab_val = np.mean(list(ref_pattern_df.loc[sub, rois_to_analyze]))
        ref_pattern_df.loc[sub, 'PUP_ROI_AB_Mean'] = avg_ab_val
        if avg_ab_val > 0.1: 
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

def get_pup_cortical_analysis_cols(roi_labels): 
    pup_cortical_rois = [] 
    for i, roi in enumerate(roi_labels): 
        if "precuneus" in roi.lower() or "superior frontal" in roi.lower() \
            or "rostral middle frontal" in roi.lower() or "lateral orbitofrontal" in roi.lower() \
            or "medial orbitofrontal" in roi.lower() or "superior temporal" in roi.lower() \
            or "middle temporal" in roi.lower(): 
            pup_cortical_rois.append(roi)
    return pup_cortical_rois

def get_clinical_status(res, clinical_df): 
    for sub in res.index: 
        visit = res.loc[sub, 'visit_label']
        cdr = clinical_df[(clinical_df.IMAGID == sub) & (clinical_df.visit == visit)].cdrglob.values[0]
        if cdr > 0:  
            res.loc[sub, 'Symptomatic'] = True 
        else: 
            res.loc[sub, 'Symptomatic'] = False
        res.loc[sub, 'CDR'] = cdr
    return res

def plot_effective_anat_dist_vs_ab(ref_pattern_df, acp_matrix, epicenters_idx, roi_labels, output_dir):
    # set diagonal to 1 
    for i in range(0, len(acp_matrix)): 
        acp_matrix[i][i] = 1
    acp_matrix_reciprocal = np.reciprocal(acp_matrix)
    acp_effective_dist = bct.distance_wei(acp_matrix_reciprocal)
    roi_labels_not_seed = []
    effective_anat_distance_dict = {}
    for i, roi in enumerate(roi_labels):
        dist = 0 
        for j in epicenters_idx: 
            dist += acp_effective_dist[0][i, j]
        dist = dist / len(epicenters_idx)
        #print("{0}: {1}".format(roi, str(np.round(dist,3))))
        effective_anat_distance_dict[roi] = dist 
    roi_dist_ab_df = pd.DataFrame(columns=["Avg_Deposition_Asymptomatic", 
                                           "Avg_Deposition_Symptomatic", 
                                           "Effective_Anat_Distance"], 
                                  index=roi_labels, 
                                  dtype="float")
    for i,roi in enumerate(roi_labels): 
        roi_dist_ab_df.loc[roi, "Effective_Anat_Distance"] = effective_anat_distance_dict[roi]
        roi_dist_ab_df.loc[roi, "Avg_Deposition_Asymptomatic"] = np.mean(ref_pattern_df[ref_pattern_df.Symptomatic == False].loc[:, roi])
        roi_dist_ab_df.loc[roi, "Avg_Deposition_Symptomatic"] = np.mean(ref_pattern_df[ref_pattern_df.Symptomatic == True].loc[:, roi])
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, sharex=False, figsize=(6,3))
    axes = [ax1, ax2]
    for i, status in enumerate(["Avg_Deposition_Asymptomatic", "Avg_Deposition_Symptomatic"]): 
        axes[i] = sns.regplot(x="Effective_Anat_Distance", y=status, data=roi_dist_ab_df.loc[roi_labels,:], ax=axes[i])

        r, p = stats.pearsonr(roi_dist_ab_df.loc[roi_labels, "Effective_Anat_Distance"],
                              roi_dist_ab_df.loc[roi_labels, status])
        title = status.split("_")[-1]
        axes[i].set_xlabel("Effective Anatomical Distance", fontsize=10, axes=axes[i])
        axes[i].set_ylabel(r"Regional A$\beta$ Probability", fontsize=10, axes=axes[i])
        axes[i].set_title(title, fontsize=10)
        axes[i].set_ylim([-0.1,1])
        axes[i].text(x=1.5, y=0.8, s="r: {0}\np < 0.01".format(str(np.round(r,3))))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "effective_anat_dist_vs_ab.png"))

def plot_clinical_status_vs_esm_params(res, output_dir):
    plt.figure(figsize=(10,10))
    nrows = 2
    ncols = 2
    params = ["BETAS_est", "DELTAS_est", "BDR_log", "PUP_ROI_AB_Mean"]
    titles = ["Production rate", "Clearance rate", "Prod/Clear rate", "Amyloid Beta"]
    for i, param in enumerate(params):  
        j = i + 1 
        plt.subplot(nrows, ncols, j)
        sns.boxplot(x="Symptomatic", y=param, data=res[res.AB_Positive == True])
        plt.title(titles[i], fontsize=12) 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clinical_status_vs_esm_params.png"))

def plot_ref_vs_pred_group_brain(ref_pattern_df, pred_pattern_df, roi_labels, output_dir): 
    dkt_atlas = nib.load("../../data/atlases/dkt_atlas_1mm.nii.gz")
    dkt_data = dkt_atlas.get_data()
    age_ranges = [[-20, -10], [-10,0], [0,10], [10,20]] 
    img_slice = [-2, 2, 24]
    n_rows = len(age_ranges)
    fig = plt.figure(figsize=(20,20))
    threshold = 0.1
    for idx, ar in enumerate(age_ranges):
        loc = 2*(idx+1)-1

        curr_ref_probs = np.zeros_like(dkt_data)
        curr_pred_probs = np.zeros_like(dkt_data)
        for i in range(0, len(roi_labels)): 
            n = len(ref_pattern_df[(ref_pattern_df.DIAN_EYO > ar[0]) & (ref_pattern_df.DIAN_EYO < ar[1])][roi_labels[i]])
            curr_ref_probs[dkt_data==(i+1)] = np.mean(ref_pattern_df[(ref_pattern_df.DIAN_EYO > ar[0]) & (ref_pattern_df.DIAN_EYO < ar[1])][roi_labels[i]])
            curr_pred_probs[dkt_data==(i+1)] = np.mean(pred_pattern_df[(pred_pattern_df.DIAN_EYO > ar[0]) & (pred_pattern_df.DIAN_EYO < ar[1])][roi_labels[i]])
        curr_ref_probs_img = nib.Nifti1Image(curr_ref_probs, affine=dkt_atlas.affine, header=dkt_atlas.header)
        curr_pred_probs_img = nib.Nifti1Image(curr_pred_probs, affine=dkt_atlas.affine, header=dkt_atlas.header)

        ax1 = plt.subplot(n_rows, 2, loc)
        plotting.plot_stat_map(curr_ref_probs_img, colorbar=True, draw_cross=False, vmax=1, cut_coords=img_slice, axes=ax1, threshold=threshold, cmap="Spectral_r")
        plt.title("EYO: {0} to {1} (n = {2})".format(str(ar[0]), str(ar[1]), str(n)), fontsize=36)
        ax2 = plt.subplot(n_rows, 2, loc+1)
        plotting.plot_stat_map(curr_pred_probs_img, colorbar=True, draw_cross=False, vmax=1, cut_coords=img_slice, axes=ax2, threshold=threshold, cmap="Spectral_r")
    fig.text(x=0.25, y=0.9,s="Observed", fontsize=36)
    fig.text(x=0.65, y=0.9,s="Predicted", fontsize=36)
    plt.savefig(os.path.join(output_dir, "ref_vs_pred_group_brain.png"))

def main(): 
    parser = ArgumentParser()
    parser.add_argument("filename",
                        help="Please pass base filename of ESM output file to analyze")
    parser.add_argument("dataset", 
                        help="Please specify whether the analysis is being done for DIAN or ADNI.")
    parser.add_argument("--scale", 
                        type=bool, 
                        default=False, 
                        help="Whether or not the input had been sigmoid normalized prior to running ESM.")

    results = parser.parse_args()
    scale = results.scale
    dataset = results.dataset

    if scale == True: 
        ref_pattern = "ref_pattern_orig"
    else: 
        ref_pattern = "ref_pattern"

    esm_output_file = "../../data/DIAN/esm_output_mat_files/" + results.filename + ".mat"
    esm_output = esm.loadmat(esm_output_file)

    ref_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=esm_output['roi_labels'], 
                                 data=esm_output[ref_pattern].transpose())
    pred_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=esm_output['roi_labels'], 
                                 data=esm_output['model_solutions0'].transpose())
    early_acc_rois = ["precuneus", "medial orbitofrontal", "posterior cingulate", "caudate", "putamen"] 
    acp_matrix = esm_output['Conn_Matrix']
    subs = esm_output['sub_ids']
    visit_labels = esm_output['visit_labels']
    roi_labels = esm_output['roi_labels']
    # change to python indexing 
    epicenters_idx = [x-1 for x in list(esm_output['seed_regions_1'][0])]
    print(epicenters_idx)
    pup_cortical_rois = get_pup_cortical_analysis_cols(roi_labels)
    ref_pattern_df = set_ab_positive(ref_pattern_df, pup_cortical_rois)

    # make a new directory for figs corresponding to a specific output? 
    output_dir = os.path.join("../../figures", results.filename)
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir)

    res = esm.Evaluate_ESM_Results(esm_output_file,
                                   sids=subs,
                                   labels=roi_labels,
                                   lit=True,
                                   plot=False)

    for i, sub in enumerate(res.index): 
        res.loc[sub, 'esm_idx'] = i

    res['visit_label'] = visit_labels
    res['BETAS_est'] = [x[0] for x in esm_output['BETAS_est']]
    res['DELTAS_est'] = [x[0] for x in esm_output['DELTAS_est']]
    res['BDR_log'] = np.log(res['BETAS_est']/res['DELTAS_est'])
    res['AB_Positive'] = ref_pattern_df['AB_Positive']
    res['PUP_ROI_AB_Mean'] = ref_pattern_df['PUP_ROI_AB_Mean']

    if dataset == "DIAN":
        genetic_df = pd.read_csv("../../data/DIAN/participant_metadata/GENETIC_D1801.csv")
        clinical_df = pd.read_csv("../../data/DIAN/participant_metadata/CLINICAL_D1801.csv")
        res['mutation_type'] = get_mutation_type(res.index, genetic_df)
        res['DIAN_EYO'] = get_eyo(res.index, res.visit_label, clinical_df)
        ref_pattern_df['DIAN_EYO'] = res['DIAN_EYO']
        pred_pattern_df['DIAN_EYO'] = res['DIAN_EYO'] 
        res = get_clinical_status(res, clinical_df)
        ref_pattern_df.loc[:, 'Symptomatic'] = res.loc[:, 'Symptomatic']
        plot_clinical_status_vs_esm_params(res, output_dir)
        plot_effective_anat_dist_vs_ab(ref_pattern_df, acp_matrix, epicenters_idx, roi_labels, output_dir)
        plot_ref_vs_pred_group_brain(ref_pattern_df, pred_pattern_df, roi_labels, output_dir)
    if dataset == "ADNI": 
        clinical_mat = pd.read_csv("../../") 

    cols_to_evaluate = list(roi_labels)
    cols_to_remove = ["thalamus", "globus pallidus"]
    for roi in roi_labels:
        for roi2 in cols_to_remove: 
            if roi2 in roi.lower():
                cols_to_evaluate.remove(roi)
    r2_ab_imp_cols = stats.pearsonr(ref_pattern_df.loc[:,cols_to_evaluate].mean(0), 
                                    pred_pattern_df.loc[:, cols_to_evaluate].mean(0))[0] ** 2 
    print("performance with excluded subcortical rois: {0}".format(np.round(r2_ab_imp_cols, 3)))   

    r2_sub = np.mean(res[res.AB_Positive == True].model_r2)
    print("ab pos sub level performance avg: {0}".format(r2_sub))                     
    
    plot_aggregate_roi_performance(ref_pattern_df, 
                                   pred_pattern_df, 
                                   roi_labels, 
                                   output_dir)

    roi_performance_hist(ref_pattern_df, pred_pattern_df, roi_labels, output_dir)
    
    if dataset == "DIAN":
        plot_aggregate_roi_performance_across_eyo(ref_pattern_df, 
                                                  pred_pattern_df, 
                                                  roi_labels, 
                                                  output_dir)
        plot_subject_performance(res, output_dir, dataset)

if __name__ == "__main__":
    main()
    
