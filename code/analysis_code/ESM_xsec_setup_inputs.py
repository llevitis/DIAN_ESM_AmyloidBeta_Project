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

def main():
    parser = ArgumentParser()
    parser.add_argument("--ab_prob_matrix_dir",
                        help="Please pass the files directory containing the PiB-PET probability matrices")
    parser.add_argument("--esm_input_file",
                        help="Please provide desired ESM input filename.")
    parser.add_argument("--connectivity_type",
                        help="Specify type of connectivity, e.g. FC or ACP")
    parser.add_argument("--epicenters_for_esm",
                        help="Please provide a list of regions to test as \
                              epicenters (all lower-case)",
                        nargs="+",
                        type=str,
                        default=None)
    results = parser.parse_args()

    ab_prob_matrix_dir = results.ab_prob_matrix_dir
    esm_input_file = results.esm_input_file
    connectivity_type = results.connectivity_type
    epicenters_for_esm = results.epicenters_for_esm
    print(epicenters_for_esm)

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
    ab_prob_all_visits_df = add_metadata_to_amyloid_df(ab_prob_all_visits_df,
                                                       genetic_df, 
                                                       clinical_df)    
    
    # extract df for subjects' first timepoint 
    ab_prob_t1 = ab_prob_all_visits_df[ab_prob_all_visits_df.visitNumber == 1]
    ab_prob_t1_mc = ab_prob_t1[ab_prob_t1.Mutation == 1]

    # get column names corresponding to ROIs
    roi_cols = ab_prob_t1_mc.columns[0:78]
    roi_cols_to_keep = [y for y in roi_cols if not all([x==0 for x in ab_prob_t1_mc[y]])]

    # get MATLAB compatible indices of ROIs to use as epicenters
    epicenters_idx = []
    for i, roi in enumerate(roi_cols_to_keep):
        if roi.lower() in epicenters_for_esm:  
            print(roi)
            epicenters_idx.append(i+1)

    # prepare inputs for ESM 
    output_dir = '../../data/DIAN/esm_input_mat_files/'
    conn_matrices = ['../../data/DIAN/connectivity_matrices/Matrix_ACP.mat', '../../data/DIAN/connectivity_matrices/Matrix_LONG.mat']
    conn_mat_names = ['Map', 'Map']
    conn_out_names = ['ACP', 'LONG']
    file_names = esm_input_file + '.mat'
    prob_matrices = {'test_data': ab_prob_t1_mc.loc[:, roi_cols]}
    ages = list(ab_prob_t1_mc.loc[:, 'VISITAGEc'])
    sub_ids = list(ab_prob_t1_mc.index)
    visit_labels = list(ab_prob_t1_mc.loc[:, 'visit'])

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
                               figure=False)

if __name__ == "__main__":
    main()