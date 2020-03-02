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
    parser.add_argument("esm_output_file",
                        help="Please pass base filename of ESM output\
                              file to analyze")

    results = parser.parse_args()
    esm_output_file = "../data/DIAN/esm_output_file" + results.esm_output_file
    esm_output = esm.loadmat(esm_output_file)
    subs = esm_output_file['sub_ids']
    visit_labels = esm_output_file['visit_labels']
    
