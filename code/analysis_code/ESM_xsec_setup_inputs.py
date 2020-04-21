#!/usr/bin/env python

import os
import glob 
import sys
import shutil 
import pdb
import re
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0,'..')
import ESM_utils as esm
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
 
def create_ab_prob_all_visits_df(file_paths, genetic_df, clinical_df, pib_df): 
    ab_prob_df_list = []
    for i, fp in enumerate(file_paths): 
        ab_curr_prob_df = pd.read_csv(file_paths[i], index_col=0)
        visit = file_paths[i].split(".")[-2].split("_")[-1]
        ab_curr_prob_df.loc[:, 'visit'] = visit
        #drop participants that did not pass QC according to PUP's PET processing
        for sub in ab_curr_prob_df.index: 
            if not ((pib_df['IMAGID'] == sub) & (pib_df['visit'] == visit)).any(): 
                ab_curr_prob_df = ab_curr_prob_df[ab_curr_prob_df.index != sub]
        ab_prob_df_list.append(ab_curr_prob_df)
    
    #concatenate all dataframes
    ab_prob_all_visits_df = pd.concat(ab_prob_df_list) 
    #add metadata to the dataframe 
    ab_prob_all_visits_df = add_metadata_to_amyloid_df(ab_prob_all_visits_df,
                                                       genetic_df, 
                                                       clinical_df)
    return ab_prob_all_visits_df

def add_metadata_to_amyloid_df(df, genetic_df, clinical_df):
    for sub in df.index: 
        sub_df = df[df.index == sub]
        visits = list(sub_df.visit)
        mutation = genetic_df[(genetic_df.IMAGID == sub)].Mutation.values[0]
        for i in range(0, len(visits)):
            visit = visits[i]
            dian_eyo = clinical_df[(clinical_df.IMAGID == sub) & (clinical_df.visit == visit)].DIAN_EYO.values
            age = clinical_df[(clinical_df.IMAGID == sub) & (clinical_df.visit == visit)].VISITAGEc.values
            if len(dian_eyo) == 0:
                print(sub + " " + visit)
            if len(dian_eyo) > 0:
                df.loc[(df.index == sub) & (df.visit == visit), "DIAN_EYO"] = dian_eyo[0]
                df.loc[(df.index == sub) & (df.visit == visit), "VISITAGEc"] = age[0]
                df.loc[(df.index == sub) & (df.visit == visit), "visitNumber"] = i + 1
                df.loc[(df.index == sub) & (df.visit == visit), "Mutation"] = mutation 
    return df

def get_rois_to_analyze(roi_colnames, rois_to_exclude): 
    roi_cols_to_exclude = [] 
    for col in roi_colnames: 
        for rte in rois_to_exclude: 
            if rte in col.lower(): 
                roi_cols_to_exclude.append(col)
    roi_cols_to_keep = [x for x in roi_cols if x not in roi_cols_to_exclude]
    return roi_cols_keep, roi_cols_to_exclude

def exclude_subcortical_rois(df, roi_cols_to_exclude): 
    df[roi_cols_to_exclude] = 0
    return df

def stripplot_subcortical_mc_nc(ab_prob_df): 
    plt.figure(figsize=(10,10))
    nrows = 2
    ncols = 2 
    subcortical_rois = ["Left Thalamus", "Left Caudate", "Left Putamen", "Left Globus Pallidus"]
    for i, roi in enumerate(subcortical_rois):  
        j = i + 1 
        plt.subplot(nrows, ncols, j)
        sns.stripplot(x="Mutation", y=roi, data=ab_prob_df, size=3)
        plt.title(roi, fontsize=12) 
        plt.ylabel("") 
        #plt.xticks(["Noncarrier", "Mutation Carrier"])
    plt.tight_layout()
    plt.savefig(os.path.join("../../figures", "mc_nc_roi_stripplot.png"))
    plt.close()



def sort_df(ab_prob_df):  
    # sort subjects
    ind_sorter = pd.DataFrame(ab_prob_df,copy=True)
    ind_sorter.loc[:,'mean'] = ab_prob_df.mean(axis=1)
    ind_order = ind_sorter.sort_values('mean',axis=0,ascending=True).index

    # column sorter
    col_sorter = pd.DataFrame(ab_prob_df,copy=True)
    col_sorter.loc['mean'] = ab_prob_df.mean(axis=0)
    col_order = col_sorter.sort_values('mean',axis=1,ascending=False).columns

    ab_prob_df_sorted = ab_prob_df.loc[ind_order, col_order]
    return ab_prob_df_sorted


def fsigmoid(x, a, b):
    # Define sigmoid function
    return 1.0 / (1.0 + np.exp(-a*(x-b)))

def zscore_mc_nc(ab_prob_df_mc, ab_prob_df_nc, roi_cols): 
    ab_prob_df_mc_zscore = ab_prob_df_mc.copy() 
    for roi in roi_cols: 
        mc_roi_vals = ab_prob_df_mc.loc[:, roi] 
        nc_roi_vals = ab_prob_df_nc.loc[:, roi]  
        mc_roi_vals_zscore = (mc_roi_vals-nc_roi_vals.mean())/nc_roi_vals.std()  
        ab_prob_df_mc_zscore.loc[:, roi] = np.absolute(mc_roi_vals_zscore) 
    scaler = MinMaxScaler()
    ab_prob_df_mc_zscore[roi_cols] = scaler.fit_transform(ab_prob_df_mc_zscore[roi_cols])
    return ab_prob_df_mc_zscore

def sigmoid_normalization(ab_prob_df): 
    '''
    For each ROI, a sigmoidal function is fit to the values across all 
    individuals to estimate the parameters of a sigmoid for this ROI. 
    The original ROI signal is rescaled by a multiple of this sigmoid 
    (1/2 the original signal + 1/2 orig * sigmoid). 
    
    ab_prob_df -- A subject x ROI matrix of AB binding probabilities (pandas DataFrame).
    '''
    # sort the original signal first  
    ab_prob_df_sorted = sort_df(ab_prob_df)
    ab_prob_df_scaled = pd.DataFrame(index=ab_prob_df_sorted.index, columns=ab_prob_df_sorted.columns)
    for roi in ab_prob_df_sorted.columns: 
        vals = ab_prob_df_sorted[roi]
        vals_idx = np.arange(0, len(vals))
        popt, pcov = curve_fit(fsigmoid, vals_idx, vals, method='dogbox', bounds=([0,0],[1, len(vals)]))
    
        x = np.linspace(0, len(vals), num=len(vals))
        y = fsigmoid(x, *popt)

        # wt1 and wt2 correspond to how much we're scaling the contribution of original
        # and rescaled signals
        wt1, wt2 = 1, 1
        vals_scaled = (wt1*y + wt2*vals) / 2
        ab_prob_df_scaled.loc[:, roi] = vals_scaled.values
    ab_prob_df_scaled = ab_prob_df_scaled.loc[ab_prob_df.index, ab_prob_df.columns]
    return ab_prob_df_scaled

def plot_roi_sub_heatmap(ab_prob_df, roi_cols):
    path = os.path.join("../../figures/roi_sub_heatmap.png")
    esm.Plot_Probabilites(ab_prob_df[roi_cols], cmap="Spectral_r", figsize=(20,10), path=path)


def main(args):
    parser = ArgumentParser()
    parser.add_argument("--ab_prob_matrix_dir",
                        help="Please pass the files directory containing the PiB-PET probability matrices")
    parser.add_argument("--esm_input_file",
                        help="Please provide desired ESM input filename.")
    parser.add_argument("--connectivity_type",
                        help="Specify type of connectivity, e.g. FC or ACP",
                        default="ACP")
    parser.add_argument("--epicenters_for_esm",
                        help="Please provide a list of regions to test as \
                              epicenters (all lower-case)",
                        nargs="+",
                        type=str,
                        default=None)
    parser.add_argument("--zscore",
                        help="Should the amyloid beta probabilities be z-scored.",
                        default=False,
                        type=bool) 
    parser.add_argument("--threshold",
                        help="Should the amyloid beta probabilities be thresholded.",
                        default=False,
                        type=bool)
    parser.add_argument("--scale", 
                        type=bool,
                        default=False,
                        help="Should the amyloid beta probabilities be within ROI sigmoid normalized.")
    parser.add_argument("--visitNumber",
                        default=1)
    results = parser.parse_args() if args is None else parser.parse_args(args)
    #results = parser.parse_args(args)

    ab_prob_matrix_dir = results.ab_prob_matrix_dir
    esm_input_file = results.esm_input_file
    connectivity_type = results.connectivity_type
    epicenters_for_esm = results.epicenters_for_esm
    zscore = results.zscore
    scale = results.scale
    threshold = results.threshold
    visitNumber = results.visitNumber

    if connectivity_type == "ACP": 
        conn_file = '../../data/DIAN/connectivity_matrices/Matrix_ACP.mat'
    elif connectivity_type == "FC":
        conn_file = '../../data/DIAN/connectivity_matrices/DIAN_FC_NC_Correlation_Matrix_Avg_ReducedConfounds.mat' 

    if scale == True: 
        esm_input_file = esm_input_file + "_scaled"

    file_paths = sorted(glob.glob(ab_prob_matrix_dir))

    pib_df = pd.read_csv("../../data/DIAN/participant_metadata/pib_D1801.csv")
    genetic_df = pd.read_csv("../../data/DIAN/participant_metadata/GENETIC_D1801.csv")
    clinical_df = pd.read_csv("../../data/DIAN/participant_metadata/CLINICAL_D1801.csv")

    ab_prob_all_visits_df = create_ab_prob_all_visits_df(file_paths, genetic_df, clinical_df, pib_df)   

    # get column names corresponding to ROIs
    roi_cols = ab_prob_all_visits_df.columns[0:78]
    roi_cols_to_keep = [y for y in roi_cols if not all([x==0 for x in ab_prob_all_visits_df[y]])]

    # get MATLAB compatible indices of ROIs to use as epicenters
    epicenters_idx = []
    for i, roi in enumerate(roi_cols_to_keep):
        if roi.lower() in epicenters_for_esm:  
            print(roi)
            epicenters_idx.append(i+1)

    #stripplot_subcortical_mc_nc(ab_prob_all_visits_df)
    
    # extract df for subjects' first timepoint for both mutation carriers and noncarriers  
    # For each region, create a null distribution from noncarriers' signal 
    # Calculate a z-score for each subject (with regards the non-carrier distribution) 
    # Take the absolute value of this z-score 
    # Normalize to 0-1
    ab_prob_mc = ab_prob_all_visits_df[ab_prob_all_visits_df.Mutation == 1]
    ab_prob_t1_mc = ab_prob_mc[ab_prob_mc.visitNumber == visitNumber]
    ab_prob_nc = ab_prob_all_visits_df[ab_prob_all_visits_df.Mutation == 0]

    if zscore == True:
        ab_prob_mc_zscore = ab_prob_mc.copy()
        ab_prob_mc_zscore = zscore_mc_nc(ab_prob_mc, ab_prob_nc, roi_cols_to_keep)
        ab_prob_t1_mc_zscore = ab_prob_mc_zscore[ab_prob_mc_zscore.visitNumber == visitNumber]

    if scale == True: 
        ab_prob_t1_mc_zscore_sigmoid = ab_prob_t1_mc_zscore.copy()
        ab_prob_t1_mc_zscore_sigmoid[roi_cols_to_keep] = sigmoid_normalization(ab_prob_t1_mc_zscore[roi_cols_to_keep]) 
    if threshold == True:
        ab_prob_t1_mc_zscore_threshold = ab_prob_t1_mc_zscore.copy()
        for col in roi_cols:
            ab_prob_t1_mc_zscore_threshold[col].values[ab_prob_t1_mc_zscore_threshold[col] < 0.15] = 0

    # prepare inputs for ESM 
    output_dir = '../../data/DIAN/esm_input_mat_files/'
    conn_matrices = [conn_file, '../../data/DIAN/connectivity_matrices/Matrix_LONG.mat']
    conn_mat_names = ['Map', 'Map']
    conn_out_names = ['ACP', 'LONG']
    file_names = esm_input_file + '.mat'
    ages = list(ab_prob_t1_mc.loc[:, 'VISITAGEc'])
    sub_ids = list(ab_prob_t1_mc.index)
    visit_labels = list(ab_prob_t1_mc.loc[:, 'visit'])

    # specify whether sigmoid normalized data is used as the test data. always include the un-normalized data.

    plot_roi_sub_heatmap(ab_prob_t1_mc_zscore, roi_cols)
    if scale == True: 
        prob_matrices = {'test_data': ab_prob_t1_mc_zscore_sigmoid.loc[:, roi_cols], 'orig_data': ab_prob_t1_mc_zscore.loc[:, roi_cols]}
    elif threshold == True: 
        prob_matrices = {'test_data': ab_prob_t1_mc_zscore_threshold.loc[:, roi_cols], 'orig_data': ab_prob_t1_mc_zscore_threshold.loc[:, roi_cols]}
    elif zscore == True: 
        prob_matrices = {'test_data': ab_prob_t1_mc_zscore.loc[:, roi_cols], 'orig_data': ab_prob_t1_mc_zscore.loc[:, roi_cols]}
    else: 
        prob_matrices = {'test_data': ab_prob_t1_mc.loc[:, roi_cols], 'orig_data': ab_prob_t1_mc.loc[:, roi_cols]}

    esm.Prepare_Inputs_for_ESM(prob_matrices, 
                               ages, 
                               output_dir,
                               file_names, 
                               conn_matrices,
                               conn_mat_names,
                               conn_out_names,
                               epicenters_idx,
                               sub_ids, 
                               visit_labels,
                               roi_cols_to_keep,
                               figure=False)

if __name__ == "__main__":
    main(sys.argv[1:])