#!/usr/bin/env python

import os
import glob 
import sys
import shutil 
import re
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt

sys.path.insert(0,'..')
import ESM_utils as esm
from scipy.optimize import curve_fit

def intersection(lst1, lst2): 
  
    # Use of hybrid method 
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3 

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

def main():
    parser = ArgumentParser()
    parser.add_argument("--ab_prob_matrix_file",
                        help="Please pass the files containing the ADNI PiB-PET probability matrix")
    parser.add_argument("--esm_input_file",
                        help="Please provide desired ESM input filename.")
    parser.add_argument("--connectivity_type",
                        help="Specify type of connectivity, e.g. FC or ACP")
    parser.add_argument("--scale", 
                        type=bool,
                        default=False,
                        help="Should the amyloid beta probabilities undergo within ROI sigmoid normalization.")
    parser.add_argument("--epicenters_for_esm",
                        help="Please provide a list of regions to test as \
                              epicenters (all lower-case)",
                        nargs="+",
                        type=str,
                        default=None)
    results = parser.parse_args()

    ab_prob_matrix_file = results.ab_prob_matrix_file
    esm_input_file = results.esm_input_file
    connectivity_type = results.connectivity_type
    epicenters_for_esm = results.epicenters_for_esm
    scale = results.scale

    if scale == True: 
        esm_input_file = esm_input_file + "_scaled"

    ab_prob_df = pd.read_csv(ab_prob_matrix_file, index_col=0)  
    
    # get column names corresponding to ROIs
    roi_cols = ab_prob_df.columns[0:78]
    roi_cols_to_keep = [y for y in roi_cols if not all([x==0 for x in ab_prob_df[y]])]

    # get MATLAB compatible indices of ROIs to use as epicenters
    epicenters_idx = []
    for i, roi in enumerate(roi_cols_to_keep):
        if roi.lower() in epicenters_for_esm:  
            print(roi)
            epicenters_idx.append(i+1)
    
    if scale == True: 
        # to-do: save orig, un-normalized df
        ab_prob_df_orig = ab_prob_df.copy()
        ab_prob_df[roi_cols_to_keep] = sigmoid_normalization(ab_prob_df[roi_cols_to_keep]) 

    # prepare inputs for ESM 
    output_dir = '../../data/ADNI/esm_input_mat_files/'
    conn_matrices = ['../../data/DIAN/connectivity_matrices/Matrix_ACP.mat', '../../data/DIAN/connectivity_matrices/Matrix_LONG.mat']
    conn_mat_names = ['Map', 'Map']
    conn_out_names = ['ACP', 'LONG']
    file_names = esm_input_file + '.mat'
    ages = list(ab_prob_df.loc[:, 'Age'])
    sub_ids = list(ab_prob_df.index)

    # specify whether sigmoid normalized data is used as the test data. always include the un-normalized data.

    if scale == True: 
        prob_matrices = {'test_data': ab_prob_df.loc[:, roi_cols], 'orig_data': ab_prob_df_orig.loc[:, roi_cols]}
    else: 
        prob_matrices = {'test_data': ab_prob_df_orig.loc[:, roi_cols], 'orig_data': ab_prob_df_orig.loc[:, roi_cols]}

    esm.Prepare_Inputs_for_ESM(prob_matrices, 
                               ages, 
                               output_dir,
                               file_names, 
                               conn_matrices,
                               conn_mat_names,
                               conn_out_names,
                               epicenters_idx,
                               sub_ids, 
                               roi_cols_to_keep,
                               figure=False)

if __name__ == "__main__":
    main()