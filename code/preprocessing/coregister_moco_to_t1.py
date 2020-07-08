#!usr/bin/env python 

import os
import sys 
import glob 
import re
import json
sys.path.insert(0, '..')
import pandas as pd 
from argparse import ArgumentParser
import numpy as np 
import nibabel as nib 
import nilearn.plotting as plotting
from nilearn.image import resample_to_img 
import nipype.interfaces.fsl as fsl


def create_transform_pet_t1w(pet_moco_summed_file, t1w_file, pet_coregistered_mat_file, pet_coregistered_file): 
    flt = fsl.FLIRT(bins=256)
    flt.inputs.in_file = pet_moco_summed_file
    flt.inputs.reference = t1w_file
    flt.inputs.output_type = "NIFTI_GZ"
    flt.inputs.out_file = pet_coregistered_file
    flt.inputs.out_matrix_file = pet_coregistered_mat_file
    flt.inputs.cost = "corratio"
    flt.inputs.searchr_x = [-90, 90]
    flt.inputs.searchr_y = [-90, 90]
    flt.inputs.searchr_z = [-90, 90]
    flt.inputs.dof = 6
    flt.inputs.interp = "trilinear"
    print(flt.cmdline)
    res = flt.run()
    
def apply_transform_pet_t1w(pet_moco_summed_file, t1w_file, pet_coregistered_mat_file, pet_coregistered_file):
    applyxfm = fsl.preprocess.ApplyXFM()
    applyxfm.inputs.in_file = pet_moco_summed_file
    applyxfm.inputs.in_matrix_file = pet_coregistered_mat_file
    applyxfm.inputs.out_file = pet_coregistered_file
    applyxfm.inputs.reference = t1w_file
    applyxfm.inputs.apply_xfm = True
    result = applyxfm.run() # doctest: +SKIP
    
def create_coregistered_qc_image(pet_coregistered_file, t1w_file):
    pet_coregistered_img = nib.load(pet_coregistered_file)
    t1w_img = nib.load(t1w_file)
    plotting.plot_roi(pet_coregistered_img, bg_img=t1w_img, draw_cross=False, alpha=0.3, output_file=pet_coregistered_file.replace('.nii.gz', '.png'))

def main(): 
    parser = ArgumentParser()
    parser.add_argument("--pup_output_dir",
                        help="Path to directory containing PUP derivatives")
    results = parser.parse_args()
    pup_output_dir = results.pup_output_dir
    
    pib_pet_moco_paths = sorted(glob.glob(pup_output_dir + "/sub-*/ses-*/*moco_pet.nii.gz"))
    t1w_paths = sorted(glob.glob(pup_output_dir + "/sub-*/ses-*/*_T1w.nii.gz"))
    pup_params_path = sorted(glob.glob(pup_output_dir + "/sub-*/ses-*/*_PUP_params.json"))
    
    if len(t1w_paths) == len(pib_pet_moco_paths):
        for i, pet_moco_file in enumerate(pib_pet_moco_paths):
            t1w_file = t1w_paths[i]
            pup_params_file = pup_params_path[i]
            
            prefix = ("_").join(pet_moco_file.split("/")[-1].split("_")[0:2])
            sub_dir = pet_moco_file.split("/")[-1].split("_")[0]
            ses_dir = pet_moco_file.split("/")[-1].split("_")[1]
            pet_moco_summed_file = os.path.join(pup_output_dir, sub_dir, ses_dir, 
                                                 prefix + "_moco_summed_pet.nii.gz")
            pet_coregistered_file = os.path.join(pup_output_dir, sub_dir, ses_dir, 
                                                 prefix + "_acq-pib_space-T1w_coregistered_pet.nii.gz")
            pet_coregistered_mat_file = os.path.join(pup_output_dir, sub_dir, ses_dir, 
                                                 prefix + "_acq-pib_space-T1w_coregistered_pet.mat")
            
            if not os.path.exists(pet_coregistered_file):
                pet_moco_img = nib.load(pet_moco_file)
                
                # Load up the PUP parameter json file
                with open(pup_params_file) as f:
                    pup_params_data = f.read()
                    pup_params_dict = json.loads(pup_params_data)
                
                # Get start and end frames for static image
                start_frame = int(pup_params_dict['sf'])
                end_frame = int(pup_params_dict['ef'])
                
                print(start_frame, end_frame)
                
                # Generate summed motion corrected PET image and save it to file
                pet_moco_summed_data = (pet_moco_img.get_data()[:,:,:,start_frame:end_frame]).sum(axis=3, dtype='float')
                pet_moco_summed_img = nib.Nifti1Image(pet_moco_summed_data, affine=pet_moco_img.affine, header=pet_moco_img.header)
                pet_moco_summed_img.to_filename(pet_moco_summed_file)
            
                # Create transform --> apply transform --> create QC image
                create_transform_pet_t1w(pet_moco_summed_file, t1w_file, pet_coregistered_mat_file, pet_coregistered_file)
                apply_transform_pet_t1w(pet_moco_summed_file, t1w_file, pet_coregistered_mat_file, pet_coregistered_file)
                create_coregistered_qc_image(pet_coregistered_file, t1w_file)
            else: 
                print("Coregistered PET file already exists for {0}".format(pet_coregistered_file))
    else:
        print("There's a mismatch between the motion corrected PET files {0} and T1w files {1}".format(len(pib_pet_moco_paths), 
                                                                                                       len(t1w_paths)))
        
        
if __name__ == "__main__":
    main()
