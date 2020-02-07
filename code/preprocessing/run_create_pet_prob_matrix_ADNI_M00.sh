#!/bin/bash 

python3.6 create_pet_probability_matrix.py --parametric_files_dir "/data1/llevitis/ADNI_Freesurfer_BIDS_output/sub-*/ses-M00/*pet*.nii.gz" \
  --dkt_atlas_dir "/data1/llevitis/ADNI_Freesurfer_BIDS_output/sub-*/ses-M00/*esm-regions*.nii.gz" \
  --ref_region_dir "/data1/llevitis/ADNI_Freesurfer_BIDS_output/sub-*/ses-M00/*ref-regions*.nii.gz" \
  --output_name "ADNI_APPIAN_Freesurfer_coregistered_voxelwise_ecdf_orig_method" \
  --dataset "ADNI" \
  --visit "M00"
