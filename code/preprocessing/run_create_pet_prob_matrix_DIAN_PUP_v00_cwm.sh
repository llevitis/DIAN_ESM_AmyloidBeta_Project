#!/bin/bash 

python3.6 create_pet_probability_matrix.py --parametric_files_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v00/*coregistered*.nii.gz" \
  --dkt_atlas_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v00/*DKT-esm-regions_space-T1w.nii.gz" \
  --ref_region_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v00/*DKT-core-wm_space-T1w.nii.gz" \
  --output_name "DIAN_PUP_coregistered_voxelwise_ecdf_orig_method_ref-cwm" \
  --dataset "DIAN" \
  --visit "v00"
