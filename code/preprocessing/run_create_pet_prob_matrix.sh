#!/bin/bash 

python3.6 create_pet_probability_matrix.py --parametric_files_dir "/data1/llevitis/DIAN/suvr_without_pvc_files/pib_pet_suvr_nii_files/*.nii.gz" \
  --dkt_atlas_dir "/data1/llevitis/DIAN/suvr_without_pvc_files/dkt_atlas_pet_space_nii/*.nii.gz" \
  --ref_region_dir "/data1/llevitis/DIAN/suvr_without_pvc_files/cerebellum_mask_pet_space_nii/*.nii.gz" \
  --dataset "DIAN" \
  --output_name "DIAN_APPIAN_SUVR_voxelwise_ecdf_orig_method" \
  --visit "v00"

