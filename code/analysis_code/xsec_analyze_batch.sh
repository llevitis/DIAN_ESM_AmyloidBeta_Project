#!/bin/bash 

esm_output_dir="../../data/DIAN/esm_output_mat_files/xsec/DIAN*epicenter*.mat"
for f in $esm_output_dir
do
  # get base of filename 
  f=$(basename -- $f)
  f_bn=$(echo $f| cut -f1 -d '.')
  if [[ "$f" == *"scaled"* ]]; then
    python ESM_xsec_analyze_outputs.py $f_bn "DIAN" --scale True 
  else 
    python ESM_xsec_analyze_outputs.py $f_bn "DIAN"
  fi
done
