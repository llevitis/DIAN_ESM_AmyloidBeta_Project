#!/bin/bash

python3.6 create_freesurfer_to_esm_region_atlases.py --pup_output_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output" \
  --freesurfer_mindboggle_df_path "/data1/llevitis/DIAN_ESM_ABeta_Analysis/data/atlases/DKT_Freesurfer_labels.csv" \
  --ref_region_label 1004 2004 5001 5002 \
  --ref_region_name "core-wm"
