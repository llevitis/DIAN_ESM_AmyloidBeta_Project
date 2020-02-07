#!/bin/bash

python3.6 create_freesurfer_to_esm_region_atlases.py --pup_output_dir "/data1/llevitis/ADNI_Freesurfer_BIDS_output" \
  --freesurfer_mindboggle_df_path "/data1/llevitis/DIAN_ESM_ABeta_Analysis/data/atlases/DKT_Freesurfer_labels.csv" \
  --ref_region_label 8 47
