#!/bin/bash 

python3.6 create_pet_probability_matrix.py --parametric_files_dir "/data1/llevitis/DIAN/appian_dvr_outputs/lp_files/*.nii.gz" \
  --dkt_atlas_dir "/data1/llevitis/DIAN/appian_dvr_outputs/dkt_files/*.nii.gz" \
  --ref_region_dir "/data1/llevitis/DIAN/appian_dvr_outputs/cerebellum_mask_files/*.nii.gz" \
  --output_name "DIAN_APPIAN_dvr_voxelwise_ecdf_orig_method_ref-cereb" \
  --dataset "DIAN" \
  --visit "v00"
