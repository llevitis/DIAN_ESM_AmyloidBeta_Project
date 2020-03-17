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
import ESM_xsec_setup_inputs
import ESM_utils as esm
from scipy.optimize import curve_fit

def intersection(lst1, lst2): 
  
    # Use of hybrid method 
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3

def main():
    parser = ArgumentParser()
    parser.add_argument("--ab_prob_matrix_dir",
                        help="Please pass the files directory containing the PiB-PET probability matrices")
    parser.add_argument("--esm_input_file",
                        help="Please provide desired ESM input filename.")
    parser.add_argument("--connectivity_type",
                        help="Specify type of connectivity, e.g. FC or ACP")
    parser.add_argument("--epicenters_for_esm",
                        help="Please provide a list of regions that were used as epicenters in xsec case.",
                        nargs="+",
                        type=str,
                        default=None)
    results = parser.parse_args()

    ab_prob_matrix_dir = results.ab_prob_matrix_dir
    esm_input_file = results.esm_input_file
    connectivity_type = results.connectivity_type
    epicenters_for_esm = results.epicenters_for_esm

    file_paths = sorted(glob.glob(ab_prob_matrix_dir))

    pib_df = pd.read_csv("../../data/DIAN/participant_metadata/pib_D1801.csv")
    genetic_df = pd.read_csv("../../data/DIAN/participant_metadata/GENETIC_D1801.csv")
    clinical_df = pd.read_csv("../../data/DIAN/participant_metadata/CLINICAL_D1801.csv")

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
    ab_prob_all_visits_df = ESM_xsec_setup_inputs.add_metadata_to_amyloid_df(ab_prob_all_visits_df,
                                                                             genetic_df, 
                                                                             clinical_df)    
    

    # get column names corresponding to ROIs
    roi_cols = ab_prob_all_visits_df.columns[0:78]
    roi_cols_to_keep = [y for y in roi_cols if not all([x==0 for x in ab_prob_all_visits_df[y]])]

    # get MATLAB compatible indices of ROIs to use as epicenters
    epicenters_idx = []
    for i, roi in enumerate(roi_cols_to_keep):
        if roi.lower() in epicenters_for_esm:  
            print(roi)
            epicenters_idx.append(i+1)

    
    # extract df for subjects' first timepoint for both mutation carriers and noncarriers  
    # For each region, create a null distribution from noncarriers' signal 
    # Calculate a z-score for each subject (with regards the non-carrier distribution) 
    # Take the absolute value of this z-score 
    # Normalize to 0-1
    ab_prob_t1_mc = ab_prob_all_visits_df[(ab_prob_all_visits_df.visitNumber == 1) & (ab_prob_all_visits_df.Mutation == 1)]
    ab_prob_t1_nc = ab_prob_all_visits_df[(ab_prob_all_visits_df.visitNumber == 1) & (ab_prob_all_visits_df.Mutation == 0)]

    ab_prob_t2_mc = ab_prob_all_visits_df[(ab_prob_all_visits_df.visitNumber == 2) & (ab_prob_all_visits_df.Mutation == 1)]
    ab_prob_t2_nc = ab_prob_all_visits_df[(ab_prob_all_visits_df.visitNumber == 2) & (ab_prob_all_visits_df.Mutation == 0)]

    ab_prob_t1_mc_zscore = ab_prob_t1_mc.copy()
    ab_prob_t1_mc_zscore = ESM_xsec_setup_inputs.zscore_mc_nc(ab_prob_t1_mc, ab_prob_t1_nc, roi_cols_to_keep)

    ab_prob_t2_mc_zscore = ab_prob_t2_mc.copy() 
    ab_prob_t2_mc_zscore = ESM_xsec_setup_inputs.zscore_mc_nc(ab_prob_t2_mc, ab_prob_t2_nc, roi_cols_to_keep)

    common_subs = sorted(intersection(list(ab_prob_t1_mc_zscore.index), list(ab_prob_t2_mc_zscore.index)))    

    # prepare inputs for ESM 
    output_dir = '../../data/DIAN/esm_input_mat_files/'
    conn_matrices = ['../../data/DIAN/connectivity_matrices/Matrix_ACP.mat', '../../data/DIAN/connectivity_matrices/Matrix_LONG.mat']
    conn_mat_names = ['Map', 'Map']
    conn_out_names = ['ACP', 'LONG']
    file_names = esm_input_file + '_zscored_longi.mat'
    ages = {'ages_v1': list(ab_prob_t1_mc_zscore.loc[common_subs, 'VISITAGEc']),
            'ages_v2': list(ab_prob_t2_mc_zscore.loc[common_subs, 'VISITAGEc'])}
    sub_ids = common_subs
    prob_matrices = {'v1': ab_prob_t1_mc_zscore.loc[common_subs, roi_cols],
                     'v2': ab_prob_t2_mc_zscore.loc[common_subs, roi_cols]}
    visit_labels = {'visit_v1': list(ab_prob_t1_mc_zscore.loc[common_subs, 'visit']),
                    'visit_v2': list(ab_prob_t2_mc_zscore.loc[common_subs, 'visit'])}    

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
    main()

