#!/bin/bash 

cbrain_adni_freesurfer_dir="ADNI_BIDS_M00-Step1-participant-943388-1"
for sub in `cat /data1/llevitis/ADNI_M00_sub_list.txt`; do 
   echo $sub
   cbrain_path=$cbrain_adni_freesurfer_dir/sub-$sub/mri/aparc+aseg.mgz
   mkdir /data1/llevitis/ADNI_Freesurfer_BIDS_output/sub-$sub 
   local_filepath="sub-"$sub"_ses-M00_aparc+aseg.mgz"
   local_abs_path="/data1/llevitis/ADNI_Freesurfer_BIDS_output/sub-"$sub"/ses-M00/"$local_filepath
   sshpass -p $SSHPASS sftp -oPort=7500 llevitis@ace-cbrain-1.cbrain.mcgill.ca:$cbrain_path $local_abs_path
done
