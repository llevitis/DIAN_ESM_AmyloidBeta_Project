#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import pdb
import pandas as pd

sys.path.insert(0,'..')
import ESM_xsec_setup_inputs

def main():
    parser = ArgumentParser()
    parser.add_argument("ref_region",
                        help="Please pass the reference region to use")
    parser.add_argument("--connectivity_type", 
                        default="ACP",
                        help="Please specify whether the connectivity type is ACP or FC")

    results = parser.parse_args()
    ref_region = results.ref_region
    conn_type = results.connectivity_type

    dkt_labels = pd.read_csv("../../data/atlases/dst_labels.csv", header=None)
    dkt_labels.columns = ["ID", "Label"]
    roi_names = sorted([x.lstrip().lower() for x in list(set(dkt_labels.Label[0:78]))])
    dkt_labels.Label[0:39] = ["left " + x.lstrip().lower() for x in dkt_labels.Label[0:39]]
    dkt_labels.Label[39:78] = ["right " + x.lstrip().lower() for x in dkt_labels.Label[39:78]]
    
    # prepare a new input file for each unique roi name
    for roi_newname in roi_names:
        if roi_newname.endswith('???'):
            roi_newname = roi_newname[:-3]  
        epicenters_for_esm = [] 
        for roi in dkt_labels.Label: 
            if roi_newname in roi: 
                epicenters_for_esm.append(roi)
        epicenters_for_esm = ",".join(epicenters_for_esm)
        if ref_region == "brainstem":
            ab_prob_matrix_dir = "../../data/DIAN/pet_probability_matrices/DIAN_PUP_coregistered_voxelwise_ecdf_orig_method_ref-brainstem_75perc_v*.csv"
        elif ref_region == "cereb":
            ab_prob_matrix_dir = "../../data/DIAN/pet_probability_matrices/DIAN_PUP_coregistered_voxelwise_ecdf_orig_method_v*.csv"
        epicenter = "".join(roi_newname.split(" "))
        esm_input_file = "DIAN_PUP_coregistered_voxelwise_ecdf_orig_method_ref-" + ref_region + "_epicenter-" + epicenter + "_conn-" + conn_type + "_zscored"
        scale="False"
        ESM_xsec_setup_inputs.main(["--ab_prob_matrix_dir", ab_prob_matrix_dir,
                                    "--esm_input_file", esm_input_file,
                                    "--connectivity_type", "ACP",
                                    "--scale", False, 
                                    "--epicenters_for_esm", epicenters_for_esm])
    
     

if __name__ == "__main__":
    main()