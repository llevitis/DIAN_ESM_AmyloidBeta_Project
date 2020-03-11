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
import ESM_xsec_analyze_outputs

from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from nilearn import input_data, image

def stacked_bar_plot_mt_epicenter(mt_epicenter_df, output_dir):
    mt_epicenter_df["total"] = mt_epicenter_df.Cortical + mt_epicenter_df.Subcortical

    #Set general plot properties
    sns.set_style("white")
    sns.set_context({"figure.figsize": (5, 5)})

    #Plot 1 - background - "total" (top) series
    sns.barplot(x =  mt_epicenter_df.index, y =  mt_epicenter_df.total, color = "red")

    #Plot 2 - overlay - "bottom" series
    bottom_plot = sns.barplot(x =  mt_epicenter_df.index, y =  mt_epicenter_df.Cortical, color = "#0000A3")


    topbar = plt.Rectangle((0,0),.5,1,fc="red", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),.5,1,fc='#0000A3',  edgecolor = 'none')

    l = plt.legend([bottombar, topbar], ['Cortical epicenter', 'Striatal epicenter'],prop={'size':14}, loc="upper right")
    l.draw_frame(False)

    #Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("")
    bottom_plot.set_xticklabels(["PSEN1", "PSEN2", "APP"])
    bottom_plot.set_xlabel("Mutation type")
    bottom_plot.set_ylabel("Number of subjects")
    plt.title("ESM Performance Across Mutation Types and Epicenters")

    #Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(16)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "compare_subs_epicenter.png")
    plt.savefig(output_path)


def main(): 
    parser = ArgumentParser()
    parser.add_argument("cortical_epicenter_file")
    parser.add_argument("subcortical_epicenter_file")
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
    
    cortical_output_file= "../../data/DIAN/esm_output_mat_files/" + results.cortical_epicenter_file + ".mat"
    subcortical_output_file= "../../data/DIAN/esm_output_mat_files/" + results.subcortical_epicenter_file + ".mat"
    cortical_esm_output = esm.loadmat(cortical_output_file)
    subcortical_esm_output = esm.loadmat(subcortical_output_file)

    roi_labels = cortical_esm_output['roi_labels']
    
    cortical_res = esm.Evaluate_ESM_Results(cortical_output_file, 
                                            sids=cortical_esm_output['sub_ids'],
                                            labels=cortical_esm_output['roi_labels'],
                                            lit=True,
                                            plot=False)

    subcortical_res = esm.Evaluate_ESM_Results(subcortical_output_file, 
                                               sids=subcortical_esm_output['sub_ids'],
                                               labels=subcortical_esm_output['roi_labels'],
                                               lit=True,
                                               plot=False)

    pup_cortical_rois = ESM_xsec_analyze_outputs.get_pup_cortical_analysis_cols(roi_labels)
    ref_pattern_df = pd.DataFrame(index=cortical_esm_output['sub_ids'], 
                                  columns=roi_labels, 
                                  data=cortical_esm_output[ref_pattern].transpose())

    ref_pattern_df = ESM_xsec_analyze_outputs.set_ab_positive(ref_pattern_df, pup_cortical_rois)

    # to-do : characterize whether the ESM run with cortical vs subcortical (striatal) epicenters did better per subject 
    # within different mutation type groups
    genetic_df = pd.read_csv("../../data/DIAN/participant_metadata/GENETIC_D1801.csv")
    clinical_df = pd.read_csv("../../data/DIAN/participant_metadata/CLINICAL_D1801.csv")
    for df in [cortical_res, subcortical_res]:
        df['mutation_type'] = ESM_xsec_analyze_outputs.get_mutation_type(df.index, genetic_df)
        df['AB_Positive'] = ref_pattern_df['AB_Positive']

   
    mt_epicenter_df = pd.DataFrame(index=list(set(cortical_res['mutation_type'])), 
                                   columns=["Subcortical", "Cortical"])
                                    
    for mt in mt_epicenter_df.index:
        s = 0
        c = 0
        for sub in cortical_res[(cortical_res.mutation_type == mt) & (cortical_res.AB_Positive == True)].index:
            if cortical_res.loc[sub, "model_r2"] > subcortical_res.loc[sub, "model_r2"]: 
                c += 1 
            else: 
                s += 1 
        mt_epicenter_df.loc[mt, 'Cortical'] = c
        mt_epicenter_df.loc[mt, 'Subcortical'] = s
    
    stacked_bar_plot_mt_epicenter(mt_epicenter_df, "../../figures/")


if __name__ == "__main__":
    main()
    

    

