#!/bin/bash 

python3.6 create_pet_probability_matrix.py --parametric_files_dir "/data1/llevitis/DIAN/dian-pet_coregistration_suvr_space-t1w/*.nii.gz" \
  --dkt_atlas_dir "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/sub-*/ses-v00/*esm-regions_space-orig_T1w.nii.gz" \
  --ref_regions 79 80 \
  --output_name "DIAN_APPIAN_FS_coregistered_voxelwise_ecdf_orig_method" \
  --dataset "DIAN" \
  --visit "v00"
