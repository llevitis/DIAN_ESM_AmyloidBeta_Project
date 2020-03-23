#!/usr/bin/env python

import os
import glob 
import sys
import shutil 
import re
import pdb

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
import ESM_xsec_setup_inputs
import ESM_xsec_analyze_outputs
import ESM_longi_setup_inputs

from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from datetime import date
import plotly.express as px

# Return a float or list of floats equal to the average change in amyloid beta in 
# the PUP ROIs between two timepoints.
def get_pup_roi_delta(df_t1, df_t2, pup_rois): 
    if len(df_t1.shape) > 1: 
        t1_t2_puproimean_delta = []
        for sub in df_t1.index:
            pup_roi_t1 = np.mean(df_t1.loc[sub, pup_rois]) 
            pup_roi_t2 = np.mean(df_t2.loc[sub, pup_rois])
            eyo_diff = df_t2.loc[sub, 'DIAN_EYO'] - df_t1.loc[sub, 'DIAN_EYO']
            pup_roi_delta = (pup_roi_t2 - pup_roi_t1) / eyo_diff 
            t1_t2_puproimean_delta.append(pup_roi_delta)
    else: 
        pup_roi_t1 = np.mean(df_t1[pup_rois]) 
        pup_roi_t2 = np.mean(df_t2[pup_rois])
        eyo_diff = df_t2['DIAN_EYO'] - df_t1['DIAN_EYO']
        pup_roi_delta = (pup_roi_t2 - pup_roi_t1) / eyo_diff 
        t1_t2_puproimean_delta = pup_roi_delta

    return t1_t2_puproimean_delta

# Create the df passed to the gantt plotting function for all individuals who have 
# 2+ visits
def create_eyo_gantt_df(ab_prob_all_visits_df, pup_rois):
    ab_prob_mc_v1_df = ab_prob_all_visits_df[(ab_prob_all_visits_df.Mutation == 1) & (ab_prob_all_visits_df.visitNumber == 1)]
    ab_prob_mc_v2_df = ab_prob_all_visits_df[(ab_prob_all_visits_df.Mutation == 1) & (ab_prob_all_visits_df.visitNumber == 2)]
    ab_prob_mc_v3_df = ab_prob_all_visits_df[(ab_prob_all_visits_df.Mutation == 1) & (ab_prob_all_visits_df.visitNumber == 3)]
    ab_prob_gantt_df = pd.DataFrame(index=list(range(0,len(ab_prob_mc_v2_df.index))), 
                                    columns=["Task", "EYO_Start", "EYO_Finish", "AB_Start", "Complete"])
    ab_prob_gantt_df = ab_prob_gantt_df.rename(index=str)    
    for i, sub in enumerate(ab_prob_mc_v2_df.index):
        i = str(i)
        ab_prob_gantt_df.loc[i, 'Task'] = sub
        ab_prob_gantt_df.loc[i, 'Mutation'] = ab_prob_mc_v1_df.loc[sub, 'Mutation']
        ab_prob_gantt_df.loc[i, 'EYO_Start'] = np.round(ab_prob_mc_v1_df.loc[sub, 'DIAN_EYO'],1)
        ab_prob_gantt_df.loc[i, 'EYO_Finish'] = np.round(ab_prob_mc_v2_df.loc[sub, 'DIAN_EYO'],1)
        ab_prob_gantt_df.loc[i, 'AB_Start'] = np.round(np.mean(ab_prob_mc_v1_df.loc[sub, pup_rois]),2)
        ab_prob_gantt_df.loc[i, 'Duration'] = ab_prob_gantt_df.loc[i, 'EYO_Finish'] - ab_prob_gantt_df.loc[i, 'EYO_Start']
        ab_prob_gantt_df.loc[i, 'Complete'] = get_pup_roi_delta(ab_prob_mc_v1_df.loc[sub,:],ab_prob_mc_v2_df.loc[sub,:], pup_rois)
    last_idx = len(ab_prob_gantt_df.index) + 1

    for i, sub in enumerate(ab_prob_mc_v3_df.index): 
        i = str(last_idx + i)
        ab_prob_gantt_df.loc[i, 'Task'] = sub
        ab_prob_gantt_df.loc[i, 'Mutation'] = ab_prob_mc_v1_df.loc[sub, 'Mutation']
        ab_prob_gantt_df.loc[i, 'EYO_Start'] = np.round(ab_prob_mc_v2_df.loc[sub, 'DIAN_EYO'],1)
        ab_prob_gantt_df.loc[i, 'EYO_Finish'] = np.round(ab_prob_mc_v3_df.loc[sub, 'DIAN_EYO'],1)
        ab_prob_gantt_df.loc[i, 'AB_Start'] = np.round(np.mean(ab_prob_mc_v1_df.loc[sub, pup_rois]),2)
        ab_prob_gantt_df.loc[i, 'Complete'] = get_pup_roi_delta(ab_prob_mc_v2_df.loc[sub,:],ab_prob_mc_v3_df.loc[sub,:], pup_rois)
        ab_prob_gantt_df.loc[i, 'Duration'] = ab_prob_gantt_df.loc[i, 'EYO_Finish'] - ab_prob_gantt_df.loc[i, 'EYO_Start']
    return ab_prob_gantt_df

def ab_eyo_gantt_plot(df, mut_status, output_dir, minn=-0.15): 
    df_sorted = df[df.Mutation == mut_status].sort_values("EYO_Start", ascending=False)
    # set colour map
    cmap = px.colors.diverging.balance

    ind0 = df_sorted.index[0]
    # import pdb; pdb.set_trace()
    fig = go.Figure()
    minn = minn
    maxx = -minn

    for idx, row in df_sorted.iterrows():

        # Potential Update: change opacity based on AB proba at start of bar
        # then, potentially, colour bars by normalized change (i.e. change / start proba)
        alpha = 0.9 # opacity
        
        # Set colour bin from value -> map
        c = int(np.floor(len(cmap)*(row.Complete - minn) / (maxx - minn)))

        # Make sure c is a valid index ... i.e. [ 0, len(cmap) ) 
        c = len(cmap) - 1 if c >= len(cmap) else c
        c = 0 if c < 0 else c

        # Only add colourbar one a single bar (its the same)
        if idx == ind0:
            marker = {"cmin": minn,
                      "cmax": maxx,
                      "showscale": True,
                      "color": cmap[c],
                      "opacity": alpha,
                      "colorscale": cmap,
                      "line":{"width": 1,
                              "color": 'DarkSlateGrey'},
                      "colorbar": {"title": "Rate of Change",
                                   "tickvals":[minn, 0, maxx]}}
        else:
            marker={"color": cmap[c],
                    "opacity": alpha,
                    "line":{"width": .8,
                            "color": 'DarkSlateGrey'},
                    "cmin": minn,
                    "cmax": maxx}

        # Add bar plot trace
        fig.add_trace(go.Bar(x0=row.Duration,
                             base=[row.EYO_Start],
                             y=[row.AB_Start],
                            #  y=[row.Task],
                             orientation='h',
                             marker=marker,
                             hovertext=row.Complete,
                            #  width=1.5))
                             width=0.02))

    if mut_status == 1: 
        mut_status = "Mutation Carriers"
    else: 
        mut_status = "Noncarriers"

    fig.update_layout(showlegend=False, plot_bgcolor='rgb(255, 255, 255)',
                      title={'text':u"A\u03B2 Deposition Change Between Visits (" + mut_status + ")",
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                      yaxis={"title": "Initial AB Deposition Probability", "tickvals": [0, 0.5, 1],
                            # },
                             "range":[-0.05, 1.05]},
                      xaxis={"title": "Estimated Years to Symptom Onset", "showgrid": False, "zeroline": False})
    fig.write_image(os.path.join(output_dir, "mc_t1_t2_gantt_plot.pdf"))

def plot_t1_t2_relationship(esm_res,output_dir):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, sharex=False, figsize=(6,3))
    axes = [ax1, ax2]
    nrows = 1
    ncols = 2
    x_items = ['pup_roi_ref_t1', 'pup_roi_ref_t2']
    y_items = ['pup_roi_ref_t2', 'pup_roi_pred_t2']
    titles = ['ref t1 vs ref t2' , 'ref t2 vs pred t2']
    for i, x_item in enumerate(x_items):
        j = i + 1 
        plt.subplot(nrows, ncols, j)
        axes[i] = sns.regplot(x=esm_res[x_item], y=esm_res[y_items[i]])
        axes[i].set_title(titles[i])
        r,p = stats.pearsonr(x=esm_res[x_item], y=esm_res[y_items[i]])
        r2 = r ** 2 
        axes[i].text(x=0.1, y=0.9, s="r2: {0}".format(str(np.round(r2, 3))))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "longi_ref.png"))
    plt.close()

# Set accumulation status.
def set_acc_status(esm_res): 
    for sub in esm_res.index: 
        if esm_res.loc[sub, 'Ref_PUP_ROI_Delta'] > 0: 
            esm_res.loc[sub, 'Accumulator'] = "Yes"
        else: 
            esm_res.loc[sub, 'Accumulator'] = "No"
    return esm_res

def plot_param_diff_acc_status(esm_res, output_dir):
    sns.set_style("whitegrid", {'axes.grid' : False})
    yaxis_labels = ["Deltas (Clearance Parameter)", "Betas (Production Parameter)", "Beta Delta Ratio (Log)"]
    plt.figure(figsize=(19,7))
    nrows = 1
    ncols = 3
    titles = ["Clearance", "Production", "Production/Clearance"]
    for i, y in enumerate(["DELTAS_est", "BETAS_est", "BDR_log"]): 
        j = i + 1 
        plt.subplot(nrows, ncols, j)
        yaxis_label = yaxis_labels[i]
        pal = {"No": "mediumblue", "Yes": "red"}
        face_pal = {"No": "cornflowerblue", "Yes": "indianred"}
        y = y
        x = "Accumulator"
        data=esm_res[esm_res.Mutation == 1]
        g = sns.boxplot(data=data, x=x, y=y, 
                        palette=face_pal, fliersize=0)
        sns.stripplot(x=x, y=y, 
                    data=data,
                    jitter=True, dodge=True, linewidth=0.5, palette=pal)
        g.set_xticklabels(["Non-accumulator", "Accumulator"], fontsize=24) 
        add_stat_annotation(g, data=data, x=x, y=y,
                            box_pairs=[("No","Yes")],
                            test='t-test_ind', text_format='star', loc='inside', verbose=2, 
                            fontsize=18)
        plt.xlabel("", fontsize=24)
        plt.ylabel("", fontsize=18)
        plt.title(titles[i], fontsize=24)
        plt.rc('xtick',labelsize=24)
        plt.rc('ytick', labelsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "param_diff_acc_status.png"))
    plt.close()

def regional_var_explained_abs_ab(v2_ref_pattern_df, v2_pred_pattern_df, roi_labels):
    r,p = stats.pearsonr(v2_ref_pattern_df[roi_labels].mean(1), v2_pred_pattern_df[roi_labels].mean(1))
    r2 = r ** 2 
    return r2

def regplot_params_vs_delta(esm_res, output_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False, sharex=False, figsize=(15,6))
    axes = [ax1, ax2, ax3]
    nrows = 1
    ncols = 3
    y_items = ['BETAS_est', 'DELTAS_est', 'BDR_log']
    titles = ['Production' , 'Clearance', 'Production/Clearance (Log)']
    for i, y_item in enumerate(y_items):
        j = i + 1 
        plt.subplot(nrows, ncols, j)
        x_pos = -0.15
        y_pos = np.max(esm_res[y_items[i]])-0.1
        axes[i] = sns.regplot(x=esm_res.Ref_PUP_ROI_Delta, y=esm_res[y_items[i]])
        axes[i].set_title(titles[i], fontsize=18)
        axes[i].set_xlabel("Delta in Cortical ROIs", fontsize=18)
        axes[i].set_ylabel("")
        r,p = stats.pearsonr(x=esm_res.Ref_PUP_ROI_Delta, y=esm_res[y_items[i]])
        if p < 0.05: 
            sig = "*" 
        else: 
            sig = "ns"
        axes[i].text(x=x_pos, y=y_pos, s="r: {0}\np: {1}".format(str(np.round(r, 3)), sig), fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regplot_params_vs_delta.png"))
    plt.close()

def main(): 
    parser = ArgumentParser()
    parser.add_argument("filename",
                        help="Please pass base filename of ESM output file to analyze")
    parser.add_argument("ab_prob_matrix_dir", 
                        help="Please pass the files directory containing the PiB-PET probability matrices")
    parser.add_argument("dataset", 
                        help="Please specify whether the analysis is being done for DIAN or ADNI.")
    results = parser.parse_args()

    # load DIAN metadata files
    pib_df = pd.read_csv("../../data/DIAN/participant_metadata/pib_D1801.csv")
    genetic_df = pd.read_csv("../../data/DIAN/participant_metadata/GENETIC_D1801.csv")
    clinical_df = pd.read_csv("../../data/DIAN/participant_metadata/CLINICAL_D1801.csv")

    ab_prob_matrix_dir = results.ab_prob_matrix_dir
    file_paths = sorted(glob.glob(ab_prob_matrix_dir))
    ab_prob_all_visits_df = ESM_xsec_setup_inputs.create_ab_prob_all_visits_df(file_paths, genetic_df, clinical_df, pib_df)
     
    esm_output_file = "../../data/DIAN/esm_output_mat_files/longi/" + results.filename + ".mat"
    esm_output = esm.loadmat(esm_output_file)
    output_dir = os.path.join("../../figures", results.filename)
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir)
    roi_labels_esm_output = [x.rstrip() for x in esm_output['roi_labels']]
    roi_labels = ab_prob_all_visits_df.columns[0:78]
    roi_labels_to_keep = [y for y in roi_labels if not all([x==0 for x in ab_prob_all_visits_df[y]])]

    pup_rois = ESM_xsec_analyze_outputs.get_pup_cortical_analysis_cols(roi_labels_esm_output)

    v1_ref_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=roi_labels_esm_output, 
                                 data=esm_output['bl_pattern'].transpose())
    v2_ref_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=roi_labels_esm_output, 
                                 data=esm_output['ref_pattern'].transpose())
    v2_pred_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=roi_labels_esm_output, 
                                 data=esm_output['model_solutions0'].transpose()) 

    v1_ref_pattern_df.loc[:, 'visit_label'] = [x for x in list(esm_output['visit_v1'][0])]                   
    v2_ref_pattern_df.loc[:, 'visit_label'] = [x for x in list(esm_output['visit_v2'][0])]
    v2_pred_pattern_df.loc[:, 'visit_label'] = v2_ref_pattern_df.loc[:, 'visit_label']

    for df in [v1_ref_pattern_df, v2_ref_pattern_df, v2_pred_pattern_df]: 
        df['DIAN_EYO'] = ESM_xsec_analyze_outputs.get_eyo(df.index, df.visit_label, clinical_df)
        df['Mutation'] = 1

    #v1_ref_pattern_df['T1_T2_PUP_ROI_Delta'] = get_pup_roi_delta(v1_ref_pattern_df, v2_ref_pattern_df, pup_rois)
    gantt_df = create_eyo_gantt_df(ab_prob_all_visits_df, pup_rois)
    ab_eyo_gantt_plot(gantt_df, mut_status=1, output_dir=output_dir, minn=-0.2)

    esm_res = pd.DataFrame(index=v1_ref_pattern_df.index, 
                           columns=["Mutation", "DIAN_EYO_t1"])

    esm_res.loc[:, 'Mutation'] = v1_ref_pattern_df.loc[:, 'Mutation']
    esm_res.loc[:, 'DIAN_EYO_t1'] = v1_ref_pattern_df.loc[:, 'DIAN_EYO']
    esm_res.loc[:, 'BETAS_est'] = list(esm_output['BETAS_est'].flatten())
    esm_res.loc[:, 'DELTAS_est'] = list(esm_output['DELTAS_est'].flatten())


    for sub in esm_res.index: 
        visit_t1 = v1_ref_pattern_df.loc[sub, 'visit_label']
        visit_t2 = v2_ref_pattern_df.loc[sub, 'visit_label']
        esm_res.loc[sub, 'visit_t1'] = visit_t1
        esm_res.loc[sub, 'visit_t2'] = visit_t2
        esm_res.loc[sub, 'pup_roi_ref_t1'] = np.mean(v1_ref_pattern_df.loc[sub, pup_rois])
        esm_res.loc[sub, 'pup_roi_ref_t2'] = np.mean(v2_ref_pattern_df.loc[sub, pup_rois])
        esm_res.loc[sub, 'pup_roi_pred_t2'] = np.mean(v2_pred_pattern_df.loc[sub, pup_rois])
        esm_res.loc[sub, 'DIAN_EYO_t2'] = v2_ref_pattern_df.loc[sub, 'DIAN_EYO']
        eyo_diff = esm_res.loc[sub, 'DIAN_EYO_t2'] - esm_res.loc[sub, 'DIAN_EYO_t1']
        esm_res.loc[sub, 'BDR_log'] = np.log(esm_res.loc[sub,'BETAS_est']/esm_res.loc[sub,'DELTAS_est'])
        esm_res.loc[sub, 'APOE'] = genetic_df[(genetic_df.IMAGID == sub)].apoe.values[0]
        esm_res.loc[sub, 'CDR_t1'] = clinical_df[(clinical_df.IMAGID == sub) & (clinical_df.visit == visit_t1)].cdrglob.values[0]
        esm_res.loc[sub, 'CDR_t2'] = clinical_df[(clinical_df.IMAGID == sub) & (clinical_df.visit == visit_t2)].cdrglob.values[0]
        ref_delta_all_rois = ((v2_ref_pattern_df.loc[sub, roi_labels_esm_output]) - (v1_ref_pattern_df.loc[sub, roi_labels_esm_output]))/eyo_diff
        pred_delta_all_rois = (v2_pred_pattern_df.loc[sub, roi_labels_esm_output] - v1_ref_pattern_df.loc[sub, roi_labels_esm_output])/eyo_diff
        r,p = stats.pearsonr(ref_delta_all_rois, pred_delta_all_rois)
        r2 = r ** 2 
        esm_res.loc[sub, 'r2_delta'] = r2
        r,p = stats.pearsonr(v2_ref_pattern_df.loc[sub, roi_labels_esm_output], v2_pred_pattern_df.loc[sub, roi_labels_esm_output])
        r2 = r ** 2 
        esm_res.loc[sub, 'r2_abs'] = r2

    esm_res.loc[:, 'Ref_PUP_ROI_Delta'] = get_pup_roi_delta(v1_ref_pattern_df, 
                                                            v2_ref_pattern_df, 
                                                            pup_rois)
    esm_res.loc[:, 'Pred_PUP_ROI_Delta'] = get_pup_roi_delta(v1_ref_pattern_df, 
                                                                   v2_pred_pattern_df, 
                                                                   pup_rois)

    esm_res = set_acc_status(esm_res)
    plot_t1_t2_relationship(esm_res, output_dir)
    plot_param_diff_acc_status(esm_res, output_dir)
    regplot_params_vs_delta(esm_res, output_dir)


    print("Summary results for all individuals\navg r2: {0}\nstd: {1}".format(np.round(np.mean(esm_res['r2_delta']),3), np.round(np.std(esm_res['r2_delta']),3)))
    print("Average within subject (abs value) variance explained: {0}".format(np.mean(esm_res.loc[:, 'r2_abs'])))

    ## set up inputs for validating the parameters using a 3rd timepoint
    common_subs_v3 = sorted(ESM_longi_setup_inputs.intersection(list(esm_res.index), list(ab_prob_all_visits_df[ab_prob_all_visits_df.visitNumber == 3].index)))
    # extract df for subjects' first timepoint for both mutation carriers and noncarriers  
    # For each region, create a null distribution from noncarriers' signal 
    # Calculate a z-score for each subject (with regards the non-carrier distribution) 
    # Take the absolute value of this z-score 
    # Normalize to 0-1

    ab_prob_mc = ab_prob_all_visits_df[ab_prob_all_visits_df.Mutation == 1]
    ab_prob_nc = ab_prob_all_visits_df[ab_prob_all_visits_df.Mutation == 0]

    ab_prob_mc_zscore = ab_prob_mc.copy()
    ab_prob_mc_zscore = ESM_xsec_setup_inputs.zscore_mc_nc(ab_prob_mc, ab_prob_nc, roi_cols_to_keep)

    ab_prob_t2_mc_zscore = ab_prob_mc_zscore[ab_prob_mc_zscore.visitNumber == 2]
    ab_prob_t3_mc_zscore = ab_prob_mc_zscore[ab_prob_mc_zscore.visitNumber == 3]


    # prepare inputs for ESM 
    output_dir = '../../data/DIAN/esm_input_mat_files/'
    conn_matrices = ['../../data/DIAN/connectivity_matrices/Matrix_ACP.mat', '../../data/DIAN/connectivity_matrices/Matrix_LONG.mat']
    conn_mat_names = ['Map', 'Map']
    conn_out_names = ['ACP', 'LONG']
    file_names = "_".join(results.filename.split("_")[:-1])  + '_validation.mat'
    ages = {'ages_v2': list(ab_prob_t2_mc_zscore.loc[common_subs_v3, 'VISITAGEc']),
            'ages_v3': list(ab_prob_t3_mc_zscore.loc[common_subs_v3, 'VISITAGEc'])}
    prob_matrices = {'v2': ab_prob_t2_mc_zscore.loc[common_subs_v3, roi_labels],
                     'v3': ab_prob_t3_mc_zscore.loc[common_subs_v3, roi_labels]}
    visit_labels = {'visit_v2': list(ab_prob_t2_mc_zscore.loc[common_subs_v3, 'visit']),
                    'visit_v3': list(ab_prob_t3_mc_zscore.loc[common_subs_v3, 'visit'])}  
    betas0_est = list(esm_output['BETAS0_est'].flatten())  
    deltas0_est = list(esm_output['DELTAS0_est'].flatten())
    epicenters_idx = [x-1 for x in list(esm_output['seed_regions_1'][0])]

    esm.Prepare_Inputs_for_ESM(prob_matrices, 
                               ages, 
                               output_dir,
                               file_names, 
                               conn_matrices,
                               conn_mat_names,
                               conn_out_names,
                               epicenters_idx,
                               common_subs_v3, 
                               visit_labels,
                               roi_labels_to_keep,
                               figure=False, 
                               betas0=betas0_est, 
                               deltas0=deltas0_est)

    # figure out how to run the esm longi validation script from here

    longi_validation_file = glob.glob("../../data/DIAN/esm_output_mat_files/longi/" + "_".join(results.filename.split("_")[:-1])  + '_validation*.mat')[0]
    longi_validation_mat = esm.loadmat(longi_validation_file)

    esm_validation_res = pd.DataFrame(index=common_subs_v3, 
                           columns=["Mutation", "DIAN_EYO_t2", "visit_t2", "visit_t3"])

    esm_validation_res.loc[:, 'Mutation'] = v1_ref_pattern_df.loc[common_subs_v3, 'Mutation']
    esm_validation_res.loc[:, 'DIAN_EYO_t2'] = v2_ref_pattern_df.loc[common_subs_v3, 'DIAN_EYO']
    esm_validation_res.loc[:, 'visit_t2'] = list(ab_prob_t2_mc_zscore.loc[common_subs_v3, 'visit'])
    esm_validation_res.loc[:, 'visit_t3'] = list(ab_prob_t3_mc_zscore.loc[common_subs_v3, 'visit'])
    esm_validation_res.loc[:, 'age_t2'] = list(ab_prob_t2_mc_zscore.loc[common_subs_v3, 'VISITAGEc'])
    esm_validation_res.loc[:, 'age_t3'] = list(ab_prob_t3_mc_zscore.loc[common_subs_v3, 'VISITAGEc'])

    v3_ref_pattern_df = pd.DataFrame(index=common_subs_v3, 
                                 columns=roi_labels_esm_output, 
                                 data=longi_validation_mat['ref_pattern'].transpose())
    v3_pred_pattern_df = pd.DataFrame(index=common_subs_v3, 
                                 columns=roi_labels_esm_output, 
                                 data=longi_validation_mat['model_solutions0'].transpose()) 

    for df in [v3_ref_pattern_df, v3_pred_pattern_df]: 
        df['visit_label'] = list(ab_prob_t3_mc_zscore.loc[common_subs_v3, 'visit'])
        df['DIAN_EYO'] = ESM_xsec_analyze_outputs.get_eyo(df.index, df.visit_label, clinical_df)

    group_validation_r,p = stats.pearsonr(v3_ref_pattern_df.mean(1), v3_pred_pattern_df.mean(1))
    print("validation group r2: {0}".format(np.round(group_validation_r ** 2), 2)) 

    esm_validation_res.loc[:, 'Ref_PUP_ROI_Delta'] = get_pup_roi_delta(v2_ref_pattern_df.loc[common_subs_v3,:] ,
                                                                       v3_ref_pattern_df.loc[common_subs_v3,:],
                                                                       pup_rois)
    esm_validation_res.loc[:, 'Pred_PUP_ROI_Delta'] = get_pup_roi_delta(v2_ref_pattern_df.loc[common_subs_v3,:], 
                                                                        v3_pred_pattern_df.loc[common_subs_v3,:], 
                                                                        pup_rois)
    esm_validation_res = set_acc_status(esm_validation_res)

    for sub in esm_validation_res.index: 
        esm_validation_res.loc[sub, 'pup_roi_ref_t2'] = np.mean(v2_ref_pattern_df.loc[sub, pup_rois])
        esm_validation_res.loc[sub, 'pup_roi_ref_t3'] = np.mean(v3_ref_pattern_df.loc[sub, pup_rois])
        esm_validation_res.loc[sub, 'pup_roi_pred_t3'] = np.mean(v3_pred_pattern_df.loc[sub, pup_rois])
        eyo_diff = esm_validation_res.loc[sub, 'age_t3'] - esm_validation_res.loc[sub, 'age_t2']
        ref_delta_all_rois = (v3_ref_pattern_df.loc[sub, roi_labels_esm_output] - v2_ref_pattern_df.loc[sub, roi_labels_esm_output])/eyo_diff
        pred_delta_all_rois = (v3_pred_pattern_df.loc[sub, roi_labels_esm_output] - v2_ref_pattern_df.loc[sub, roi_labels_esm_output])/eyo_diff
        r,p = stats.pearsonr(ref_delta_all_rois, pred_delta_all_rois)
        r2 = r ** 2 
        esm_validation_res.loc[sub, 'r2_delta'] = r2
        esm_validation_res.loc[sub, 'T1-T2_Delta'] = esm_res.loc[sub, 'Ref_PUP_ROI_Delta']
        esm_validation_res.loc[sub, 'T1_T2_Acc_Status'] = esm_res.loc[sub, 'Accumulator']
        if esm_validation_res.loc[sub, 'T1_T2_Acc_Status'] == "Yes" and esm_validation_res.loc[sub, 'Accumulator'] == "Yes": 
            esm_validation_res.loc[sub, 't1-t3_acc_status'] = "A-A"
        elif esm_validation_res.loc[sub, 'T1_T2_Acc_Status'] == "Yes" and esm_validation_res.loc[sub, 'Accumulator'] == "No": 
            esm_validation_res.loc[sub, 't1-t3_acc_status'] = "A-NA"
        elif esm_validation_res.loc[sub, 'T1_T2_Acc_Status'] == "No" and esm_validation_res.loc[sub, 'Accumulator'] == "Yes": 
            esm_validation_res.loc[sub, 't1-t3_acc_status'] = "NA-A"
        else: 
            esm_validation_res.loc[sub, 't1-t3_acc_status'] = "NA-NA"

    pdb.set_trace()
    sns.boxplot(x="t1-t3_acc_status", y="r2_delta", data=esm_validation_res) 
    plt.show()
        

if __name__ == "__main__":
    main()