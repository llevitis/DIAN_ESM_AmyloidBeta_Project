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
import ESM_xsec_analyze_outputs

from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from datetime import date
import plotly.express as px

def get_pup_roi_delta(v1_ref_pattern_df, v2_ref_pattern_df, pup_rois): 
    t1_t2_puproimean_delta = []
    for sub in v1_ref_pattern_df.index:
        pup_roi_t1 = np.mean(v1_ref_pattern_df.loc[sub, pup_rois]) 
        pup_roi_t2 = np.mean(v2_ref_pattern_df.loc[sub, pup_rois])
        eyo_diff = v2_ref_pattern_df.loc[sub, 'DIAN_EYO'] - v1_ref_pattern_df.loc[sub, 'DIAN_EYO']
        pup_roi_delta = (pup_roi_t2 - pup_roi_t1) / eyo_diff 
        t1_t2_puproimean_delta.append(pup_roi_delta)
    return t1_t2_puproimean_delta


def create_eyo_gantt_df(v1_ref_pattern_df, v2_ref_pattern_df, roi_labels):
    ab_prob_gantt_df = pd.DataFrame(index=v1_ref_pattern_df.index, columns=["Task", "Start", "Finish", "Complete"])
    for sub in v1_ref_pattern_df.index:
        ab_prob_gantt_df.loc[sub, 'Task'] = sub
        ab_prob_gantt_df.loc[sub, 'Mutation'] = v1_ref_pattern_df.loc[sub, 'Mutation']
        ab_prob_gantt_df.loc[sub, 'Start'] = np.round(v1_ref_pattern_df.loc[sub, 'DIAN_EYO'],1)
        ab_prob_gantt_df.loc[sub, 'Finish'] = np.round(v2_ref_pattern_df.loc[sub, 'DIAN_EYO'],1)
        ab_prob_gantt_df.loc[sub, 'Complete'] = v1_ref_pattern_df.loc[sub, 'T1_T2_PUP_ROI_Delta']
        ab_prob_gantt_df.loc[sub, 'Duration'] = ab_prob_gantt_df.loc[sub, 'Finish'] - ab_prob_gantt_df.loc[sub, 'Start']
    return ab_prob_gantt_df

def ab_eyo_gantt_plot(df, mut_status, output_dir, minn=-0.15): 
    df_sorted = df[df.Mutation == mut_status].sort_values("Start", ascending=False)
    # set colour map
    cmap = px.colors.diverging.balance

    ind0 = df_sorted.index[0]

    fig = go.Figure()
    minn = minn
    maxx = -minn

    for idx, row in df_sorted.iterrows():

        # Potential Update: change opacity based on AB proba at start of bar
        # then, potentially, colour bars by normalized change (i.e. change / start proba)
        alpha = 1 # opacity
        
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
                      "colorscale": cmap}
        else:
            marker={"color": cmap[c],
                    "opacity": alpha,
                    "cmin": minn,
                    "cmax": maxx}

        # Add bar plot trace
        fig.add_trace(go.Bar(x0=row.Duration,
                             base=[row.Start],
                             y=[row.Task],
                             orientation='h',
                             marker=marker,
                             hovertext=row.Complete,
                             width=1.5))

    if mut_status == 1: 
        mut_status = "Mutation Carriers"
    else: 
        mut_status = "Noncarriers"

    fig.update_layout(showlegend=False, plot_bgcolor='rgb(255,255,255)',
                      title={'text':u"A\u03B2 Deposition Change Between Visits (" + mut_status + ")",
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                      yaxis={"title": "Subject", "tickvals": []},
                      xaxis={"title": "Estimated Years to Symptom Onset"})
    fig.write_image(os.path.join(output_dir, "mc_t1_t2_gantt_plot.pdf"))

def plot_t1_t2_relationship(esm_res):
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
    plt.savefig("../../figures/longi_ref.png")

def set_acc_status(esm_res): 
    for sub in esm_res.index: 
        if esm_res.loc[sub, 'T1_T2_Ref_PUP_ROI_Delta'] > 0: 
            esm_res.loc[sub, 'Accumulator'] = True
        else: 
            esm_res.loc[sub, 'Accumulator'] = False
    return esm_res

def plot_param_diff_acc_status(esm_res, output_dir):
    sns.set_style("whitegrid", {'axes.grid' : False})
    yaxis_labels = ["Deltas (Clearance Parameter)", "Betas (Production Parameter)", "Beta Delta Ratio (Log)"]
    plt.figure(figsize=(19,5))
    nrows = 1
    ncols = 3
    titles = ["Clearance", "Production", "Production/Clearance"]
    for i, y in enumerate(["DELTAS_est", "BETAS_est", "BDR_log"]): 
        j = i + 1 
        plt.subplot(nrows, ncols, j)
        yaxis_label = yaxis_labels[i]
        pal = {False: "mediumblue", True: "red"}
        face_pal = {False: "cornflowerblue", True: "indianred"}
        y = y
        x = "Accumulator"
        data=esm_res[esm_res.Mutation == 1]
        g = sns.boxplot(data=data, x=x, y=y, 
                        palette=face_pal, fliersize=0)
        sns.stripplot(x=x, y=y, 
                    data=data,
                    jitter=True, dodge=True, linewidth=0.5, palette=pal)
        g.set_xticklabels(["Non-accumulator", "Accumulator"], fontsize=18) 
        #add_stat_annotation(g, data=data, x=x, y=y,
        #                    box_pairs=[((False,True))],
        #                    test='t-test_ind', text_format='star', loc='inside', verbose=2, 
        #                    fontsize=24)
        plt.xlabel("", fontsize=24)
        plt.ylabel("", fontsize=18)
        plt.title(titles[i], fontsize=18)
        plt.rc('xtick',labelsize=15)
        plt.rc('ytick', labelsize=15)
    plt.savefig(os.path.join(output_dir, "param_diff_acc_status.png"))


def main(): 
    parser = ArgumentParser()
    parser.add_argument("filename",
                        help="Please pass base filename of ESM output file to analyze")
    parser.add_argument("dataset", 
                        help="Please specify whether the analysis is being done for DIAN or ADNI.")
    results = parser.parse_args()

    # load DIAN metadata files
    genetic_df = pd.read_csv("../../data/DIAN/participant_metadata/GENETIC_D1801.csv")
    clinical_df = pd.read_csv("../../data/DIAN/participant_metadata/CLINICAL_D1801.csv")
     
    esm_output_file = "../../data/DIAN/esm_output_mat_files/longi/" + results.filename + ".mat"
    esm_output = esm.loadmat(esm_output_file)
    output_dir = os.path.join("../../figures", results.filename)
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir)
    roi_labels = esm_output['roi_labels']
    pup_rois = ESM_xsec_analyze_outputs.get_pup_cortical_analysis_cols(roi_labels)

    v1_ref_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=esm_output['roi_labels'], 
                                 data=esm_output['bl_pattern'].transpose())
    v2_ref_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=esm_output['roi_labels'], 
                                 data=esm_output['ref_pattern'].transpose())
    v2_pred_pattern_df = pd.DataFrame(index=esm_output['sub_ids'], 
                                 columns=esm_output['roi_labels'], 
                                 data=esm_output['model_solutions0'].transpose()) 

    v1_ref_pattern_df.loc[:, 'visit_label'] = [x for x in list(esm_output['visit_v1'][0])]                   
    v2_ref_pattern_df.loc[:, 'visit_label'] = [x for x in list(esm_output['visit_v2'][0])]
    v2_pred_pattern_df.loc[:, 'visit_label'] = v2_ref_pattern_df.loc[:, 'visit_label']

    for df in [v1_ref_pattern_df, v2_ref_pattern_df, v2_pred_pattern_df]: 
        df['DIAN_EYO'] = ESM_xsec_analyze_outputs.get_eyo(df.index, df.visit_label, clinical_df)
        df['Mutation'] = 1

    v1_ref_pattern_df['T1_T2_PUP_ROI_Delta'] = get_pup_roi_delta(v1_ref_pattern_df, v2_ref_pattern_df, pup_rois)
    gantt_df = create_eyo_gantt_df(v1_ref_pattern_df, v2_ref_pattern_df, roi_labels)
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
        ref_delta_all_rois = ((v2_ref_pattern_df.loc[sub, roi_labels]) - (v1_ref_pattern_df.loc[sub, roi_labels]))/eyo_diff
        pred_delta_all_rois = (v2_pred_pattern_df.loc[sub, roi_labels] - v1_ref_pattern_df.loc[sub, roi_labels])/eyo_diff
        r,p = stats.pearsonr(ref_delta_all_rois, pred_delta_all_rois)
        r2 = r ** 2 
        esm_res.loc[sub, 'r2_delta'] = r2

    esm_res.loc[:, 'T1_T2_Ref_PUP_ROI_Delta'] = get_pup_roi_delta(v1_ref_pattern_df, 
                                                                  v2_ref_pattern_df, 
                                                                  pup_rois)
    esm_res.loc[:, 'T1_T2_Pred_PUP_ROI_Delta'] = get_pup_roi_delta(v1_ref_pattern_df, 
                                                                   v2_pred_pattern_df, 
                                                                   pup_rois)

    esm_res = set_acc_status(esm_res)
    plot_t1_t2_relationship(esm_res)
    plot_param_diff_acc_status(esm_res, output_dir)

    print("Summary results for all individuals\navg r2: {0}\nstd: {1}".format(np.mean(esm_res['r2_delta']), np.std(esm_res['r2_delta'])))
    print("Summary results for individuals with positive delta\navg r2: {0}\nstd: {1}".format(np.mean(esm_res[esm_res.Accumulator==True].r2_delta), np.std(esm_res[esm_res.Accumulator==True].r2_delta)))

    
    
if __name__ == "__main__":
    main()