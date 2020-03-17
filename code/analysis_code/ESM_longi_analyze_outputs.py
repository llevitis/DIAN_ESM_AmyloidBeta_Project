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

def get_pup_roi_delta(v1_ref_pattern_df, v2_ref_pattern_df, roi_labels): 
    pup_rois = ESM_xsec_analyze_outputs.get_pup_cortical_analysis_cols(roi_labels)
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
    
    esm_output_file = "../../data/DIAN/esm_output_mat_files/" + results.filename + ".mat"
    esm_output = esm.loadmat(esm_output_file)
    output_dir = os.path.join("../../figures", results.filename)
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir)
    roi_labels = esm_output['roi_labels']

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

    for df in [v1_ref_pattern_df, v2_ref_pattern_df]: 
        df['DIAN_EYO'] = ESM_xsec_analyze_outputs.get_eyo(df.index, df.visit_label, clinical_df)
        df['Mutation'] = 1

    v1_ref_pattern_df['T1_T2_PUP_ROI_Delta'] = get_pup_roi_delta(v1_ref_pattern_df, v2_ref_pattern_df, roi_labels)
    gantt_df = create_eyo_gantt_df(v1_ref_pattern_df, v2_ref_pattern_df, roi_labels)
    ab_eyo_gantt_plot(gantt_df, mut_status=1, output_dir=output_dir, minn=-0.2)

    
if __name__ == "__main__":
    main()