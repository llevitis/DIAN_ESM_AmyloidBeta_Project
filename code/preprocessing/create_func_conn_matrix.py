import os 
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn import plotting
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import json
from nilearn.connectome import ConnectivityMeasure


def main():  
    bold_files = sorted("../../data/DIAN/fmriprep_output/preproc_bold_scans/*")
    confound_files = sorted("../../data/DIAN/fmriprep_output/confound_files/*")

    genetic_df = pd.read_csv("../../data/DIAN/participant_metadata/GENETIC_D1801.csv")
    


if __name__ == "__main__":
    main()