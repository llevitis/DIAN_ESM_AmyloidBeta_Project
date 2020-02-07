#!usr/bin/env python 

import os
import sys 
import glob 
import re
sys.path.insert(0, '..')
import pandas as pd 
from argparse import ArgumentParser
import numpy as np 
import nibabel as nib 
import nilearn.plotting as plotting
from nilearn.image import resample_to_img 

def create_atlas(freesurfer_label_path, 
                 t1w_path, 
                 FS_MB_df, 
                 new_atlas_path,
                 labels_to_use): 
    t1w_image = nib.load(t1w_path)
    freesurfer_label_img = nib.load(freesurfer_label_path)
    freesurfer_label_data = freesurfer_label_img.get_data()
    new_atlas_data = np.zeros_like(freesurfer_label_data) 
    for region in labels_to_use: 
        new_atlas_data[freesurfer_label_data==region] = FS_MB_df[FS_MB_df.Freesurfer_label == region].label 
    new_atlas_img = nib.Nifti1Image(new_atlas_data, affine=freesurfer_label_img.affine, header=freesurfer_label_img.header)
    new_atlas_img = resample_to_img(new_atlas_img, t1w_image, interpolation="nearest")
    #al_t1 = nib.Nifti1Image(t1w_image.get_data(), affine=np.eye(4), header=t1w_image.header)
    #al_dkt = nib.Nifti1Image(new_atlas_img.get_data(), affine=np.eye(4), header=new_atlas_img.header)
    nib.save(new_atlas_img, new_atlas_path)
    print("New Mindboggle DKT atlas: " + new_atlas_path) 
    #plotting.plot_roi(new_atlas_img, bg_img=t1w_image, output_file=new_atlas_path.replace('.nii.gz', '.png'))
    #plotting.plot_roi(al_dkt, bg_img=al_t1, output_file=new_atlas_path.replace('.nii.gz', '_voxel_space.png'))

def create_atlas_with_esm_regions(freesurfer_label_path,
                                  t1w_path,
                                  FS_MB_df):
    new_dkt_esm_regions_atlas_path = freesurfer_label_path.replace('parcellation-DKT', 'parcellation-DKT-esm-regions')
    labels_to_use = list(FS_MB_df.Freesurfer_label)
    if not os.path.exists(new_dkt_esm_regions_atlas_path):
        create_atlas(freesurfer_label_path, t1w_path, FS_MB_df, new_dkt_esm_regions_atlas_path, labels_to_use)
    else:
        print("Found a DKT atlas in native space at: " + new_dkt_esm_regions_atlas_path) 
    

def create_atlas_with_ref_region(freesurfer_label_path,
                                 t1w_path,
                                 FS_MB_df,
                                 ref_region_label, 
                                 ref_region_name): 
    new_dkt_ref_regions_atlas_path = freesurfer_label_path.replace('parcellation-DKT', 'parcellation-DKT-' + ref_region_name)
    if not os.path.exists(new_dkt_ref_regions_atlas_path):  
        create_atlas(freesurfer_label_path, t1w_path, FS_MB_df, new_dkt_ref_regions_atlas_path, ref_region_label)
    else:
        print("Found a ref region atlas in native space at: " + new_dkt_ref_regions_atlas_path)


def main(): 
    parser = ArgumentParser()
    parser.add_argument("--pup_output_dir",
                        help="Path to directory containing PUP derivatives")
    parser.add_argument("--freesurfer_mindboggle_df_path",
                        help="Path to merged Freesurfer/Mindboggle label dataframe")
    parser.add_argument("--ref_region_label", nargs="+", type=int,
                        help="Freesurfer labels for reference region for subsequent intensity normalization")
    parser.add_argument("--ref_region_name", 
                        help="Name to use for reference region")

    args = parser.parse_args() 
    pup_output_dir = args.pup_output_dir
    freesurfer_mindboggle_df_path = args.freesurfer_mindboggle_df_path
    ref_region_label = args.ref_region_label 
    ref_region_name = args.ref_region_name
    
    print(ref_region_label)
    freesurfer_label_paths = sorted(glob.glob(pup_output_dir + "/sub-*/ses-*/*DKT_space-T1w.nii.gz"))
    t1w_paths = sorted(glob.glob(pup_output_dir + "/sub-*/ses-*/*_T1w.nii.gz"))
    

    freesurfer_mindboggle_df = pd.read_csv(freesurfer_mindboggle_df_path, index_col=0)
    
    for idx in range(0, len(freesurfer_label_paths)):
        freesurfer_label_path = freesurfer_label_paths[idx]
        t1w_path = t1w_paths[idx]
        create_atlas_with_esm_regions(freesurfer_label_path, t1w_path, freesurfer_mindboggle_df)
        create_atlas_with_ref_region(freesurfer_label_path, t1w_path, freesurfer_mindboggle_df, ref_region_label, ref_region_name) 
        

if __name__ == "__main__":
    main()       
