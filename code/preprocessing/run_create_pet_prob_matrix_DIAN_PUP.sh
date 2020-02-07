#!/bin/bash 

python3.6 create_pet_probability_matrix.py --parametric_files_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v00/*pib*.nii.gz" \
  --dkt_atlas_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v00/*esm-regions*.nii.gz" \
  --ref_region_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v00/*ref-regions*.nii.gz" \
  --output_name "DIAN_PUP_coregistered_voxelwise_ecdf_orig_method" \
  --dataset "DIAN" \
  --visit "v00"
