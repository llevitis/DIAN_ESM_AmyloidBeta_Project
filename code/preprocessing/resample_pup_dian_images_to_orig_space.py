#!/usr/bin/env python

from nipype.interfaces import fsl
import nilearn.plotting as plotting
import nibabel as nib
from nilearn.image import resample_img
import fnmatch 
import os
from glob import glob

def main():  
    pet_coreg_files = sorted(glob("/data1/llevitis/DIAN/dian-pet_coregistration_suvr_space-t1w/*.nii.gz"))
    dian_bids_dir = "/home/users/llevitis/ace_mount/ace_home/DIAN/Nifti/"
    dian_pup_dir = "/home/users/llevitis/ace_mount/ace_home/DIAN_PUP_output/"
    for pcf in pet_coreg_files: 
        sub = pcf.split("/")[-1].split("_")[0].split("-")[1] 
        raw_t1w_v00_file = glob(os.path.join(dian_bids_dir, "sub-" + sub, "ses-v00", "anat/", "*_T1w.nii.gz"))[0]
        print(raw_t1w_v00_file)
        pup_t1w_file = glob(os.path.join(dian_pup_dir, "sub-" + sub, "ses-v00", "*ses-v00_T1w.nii.gz"))
        if len(pup_t1w_file) == 1: 
            print(sub)
            pup_t1w_file = pup_t1w_file[0]
            pup_fs_file = glob(os.path.join(dian_pup_dir, "sub-" + sub, "ses-v00", "*_parcellation-DKT-esm-regions*.nii.gz"))[0]

            pup_t1w_file_flipped = pup_t1w_file.replace("_T1w.nii.gz", "_T1w_flipped.nii.gz") 
            pup_fs_file_flipped = pup_fs_file.replace("_space-T1w.nii.gz", "_space-T1w_flipped.nii.gz")

            print(pup_fs_file_flipped)

            flt_out_file = pup_t1w_file.replace("_T1w.nii.gz", "_T1w_space-orig-T1w.nii.gz")

            # flip the pup images since they're for whatever reason oriented the opposite way
            # just registering them without flipping them doesn't do a perfect job 

            pup_t1w_img = nib.load(pup_t1w_file)
            pup_t1w_aff = pup_t1w_img.affine
            pup_t1w_aff[0,0] *= -1
            pup_t1w_img = resample_img(pup_t1w_img, target_affine=pup_t1w_aff, interpolation='nearest')
            nib.save(pup_t1w_img, pup_t1w_file_flipped)	

            pup_fs_img = nib.load(pup_fs_file)
            pup_fs_aff = pup_fs_img.affine
            pup_fs_aff[0,0] *= -1
            pup_fs_img = resample_img(pup_fs_img, target_affine=pup_fs_aff, interpolation='nearest')
            nib.save(pup_fs_img, pup_fs_file_flipped)

            pup_fs_file_resampled = pup_fs_file.replace("_space-T1w.nii.gz", "_space-orig_T1w.nii.gz")


            if not os.path.exists(pup_fs_file_resampled):
                flt = fsl.FLIRT(bins=256)
                flt.inputs.in_file = pup_t1w_file_flipped
                flt.inputs.reference = raw_t1w_v00_file
                flt.inputs.output_type = "NIFTI_GZ"
                flt.inputs.out_file = flt_out_file
                flt.inputs.out_matrix_file = pup_t1w_file.replace("_T1w.nii.gz", "_space-orig_T1w_xfm.mat")
                flt.inputs.cost = "corratio"
                flt.inputs.searchr_x = [-90, 90]
                flt.inputs.searchr_y = [-90, 90]
                flt.inputs.searchr_z = [-90, 90]
                flt.inputs.dof = 6
                flt.inputs.interp = "trilinear"
                print(flt.cmdline)
                res = flt.run()
            
                # apply the transform 
                applyxfm = fsl.ApplyXFM() 
                applyxfm.inputs.in_file = pup_fs_file_flipped
                applyxfm.inputs.reference = raw_t1w_v00_file
                applyxfm.inputs.out_file = pup_fs_file.replace("_space-T1w.nii.gz", "_space-orig_T1w.nii.gz")
                applyxfm.inputs.in_matrix_file = pup_t1w_file.replace("_T1w.nii.gz", "_space-orig_T1w_xfm.mat")
                applyxfm.inputs.apply_xfm = True
                applyxfm.inputs.interp = "nearestneighbour"
                print(applyxfm.cmdline)
                res = applyxfm.run()

if __name__ == "__main__":
    main()
