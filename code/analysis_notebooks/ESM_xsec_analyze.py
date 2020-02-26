import os
import glob 
import shutil 
import re

import nibabel as nib

import create_pet_probability_matrix

import pandas as pd
import numpy as np
import nilearn.plotting as plotting
import itertools
import matplotlib.colors as colors
import seaborn as sns

import matplotlib.pyplot as plt
import math
from statannot import add_stat_annotation

from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from nilearn import input_data, image

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



#def main(prob_matrix, esm_output_file):



