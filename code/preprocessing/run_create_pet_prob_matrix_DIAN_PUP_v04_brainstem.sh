#!/bin/bash 

python3.6 create_pet_probability_matrix.py --parametric_files_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v04/*coregistered*.nii.gz" \
  --dkt_atlas_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v04/*DKT-esm-regions_space-T1w.nii.gz" \
  --ref_region_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v04/*DKT-brainstem_space-T1w.nii.gz" \
  --output_name "DIAN_PUP_coregistered_voxelwise_ecdf_orig_method_ref-brainstem" \
  --dataset "DIAN" \
  --visit "v04"
