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


def plot_aggregate_roi_performance(esm_output, output_dir): 
    plt.figure(figsize=(5,5))
    sns.regplot(esm_output['ref_pattern'].mean(1), esm_output['model_solutions0'].mean(1), color="indianred")
    r,p = stats.pearsonr(esm_output['ref_pattern'].mean(1), esm_output['model_solutions0'].mean(1))
    r2 = r ** 2 
    xmin = np.min(esm_output['ref_pattern'].mean(1))
    ymax = np.max(esm_output['model_solutions0'].mean(1))
    plt.text(xmin, ymax, "$r^2$ = {0}".format(np.round(r2, 2), fontsize=16))
    plt.xticks(x=16)
    plt.yticks(y=16)
    plt.xlabel(r"Observed A$\beta$ Probabilities", fontsize=16)
    plt.ylabel(r"Predicted A$\beta$ Probabilities", fontsize=16)
    plt.title(r"Average A$\beta$ Pattern Across All MC", fontsize=16)
    output_path = os.path.join(output_dir, "aggreggate_roi_performance_xsec.png")
    plt.tight_layout()
    plt.savefig(output_path)


def main(): 
    parser = ArgumentParser()
    parser.add_argument("filename",
                        help="Please pass base filename of ESM output file to analyze")

    results = parser.parse_args()
    esm_output_file = "../../data/DIAN/esm_output_mat_files/" + results.filename + ".mat"
    esm_output = esm.loadmat(esm_output_file)
    subs = esm_output['sub_ids']
    visit_labels = esm_output['visit_labels']
    roi_labels = esm_output['roi_labels']

    # make a new directory for figs corresponding to a specific output? 
    output_dir = os.path.join("../../figures", results.filename)
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir)
    
    plot_aggregate_roi_performance(esm_output, output_dir)

    res = esm.Evaluate_ESM_Results(esm_output_file,
                                   sids=subs,
                                   labels=roi_labels,
                                   lit=True,
                                   plot=False)

if __name__ == "__main__":
    main()
    
