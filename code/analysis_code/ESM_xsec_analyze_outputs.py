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

def main(): 
    parser = ArgumentParser()
    parser.add_argument("filename",
                        help="Please pass base filename of ESM output file to analyze")

    results = parser.parse_args()
    esm_output_file = "../../data/DIAN/esm_output_mat_files/" + results.filename
    esm_output = esm.loadmat(esm_output_file)
    subs = esm_output['sub_ids']
    visit_labels = esm_output['visit_labels']
    roi_labels = esm_output['roi_labels']

    plt.figure(1)
    sns.regplot(esm_output['ref_pattern'].mean(1), esm_output['model_solutions0'].mean(1)) 
    plt.title("Reference vs Predicted Average Pattern Across Regions", fontsize=18)
    plt.xlabel("Reference", fontsize=18)
    plt.ylabel("Predicted", fontsize=18)

    plt.figure(2)    
    sns.regplot(esm_output['ref_pattern'].mean(0), esm_output['model_solutions0'].mean(0)) 
    plt.title("Reference vs Predicted Average Pattern Across Subjects", fontsize=18)
    plt.show()

    res = esm.Evaluate_ESM_Results(esm_output_file,
                                   sids=subs,
                                   labels=roi_labels,
                                   lit=True,
                                   plot=False)


if __name__ == "__main__":
    main()
    
