#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:22:16 2018

@author: tpisano
"""
import os, numpy as np, pandas as pd, cv2, skimage, sys, shutil, multiprocessing as mp, time
import subprocess as sp
from skimage.external import tifffile
import matplotlib.pyplot as plt
import cPickle as pickle
#sudo apt-get install openslide-tools
#pip install openslide-python
#https://openslide.org/api/python/
#https://openslide.org/formats/hamamatsu/
import openslide

def listdirfull(src, keyword=False):
    if not keyword: fls = [os.path.join(src, xx) for xx in os.listdir(src) if 'Thumbs.db' not in xx]
    if keyword: fls = [os.path.join(src, xx) for xx in os.listdir(src) if 'Thumbs.db' not in xx and keyword in xx]
    fls.sort()
    return fls

def save_dictionary(dst, dct):
    '''Basically the same as save_kwargs
    '''
    if dst[-2:]!='.p': dst=dst+'.p'

    with open(dst, 'wb') as fl:
        pickle.dump(dct, fl, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_dictionary(pth):
    '''simple function to load dictionary given a pth
    '''
    kwargs = {};
    with open(pth, 'rb') as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()

    return kwargs

def makedir(src):
    if not os.path.exists(src): os.mkdir(src)
    return
def removedir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    return

def rescale_pnts(pntsfld, outfld, inlevel=6, outlevel=2):
    '''Simple function to rescale marked points if marking at one level but want to use another level.
    
    pntsfld = location of previously marked points
    outfld = new location to store new points
    inlevel = level of marked points
    outlevel = desired level
    '''	
    #setup
    makedir(outfld)
    
    #get levels
    #fl = openslide.OpenSlide(src);dims = fl.level_dimensions;inn = dims[inlevel-1];out = dims[outlevel-1];scale = [b/float(a) for a,b in zip(inn,out)][0]
    scale = 2**(inlevel-outlevel)
    
    #rescale
    for fl in listdirfull(pntsfld, '.p'):
        dct = load_dictionary(fl)
        ndct = {k:[[yy*scale for yy in xx] for xx in v] for k,v in dct.iteritems()}
        save_dictionary(os.path.join(outfld, os.path.basename(fl)), ndct)
        
    return
#%%

if __name__ == '__main__':
    pntsfld = '/media/tpisano/FAT32/user_defined_points'
    outfld = '/media/tpisano/FAT32/user_defined_points_transformed'
    inlevel=6
    outlevel=2
    rescale_pnts(pntsfld, outfld, inlevel, outlevel)
    
    src = ['/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-1.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-2.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-3 - 2018-07-18 18.04.40.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-4 - 2018-07-18 04.14.19.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-5 - 2018-07-18 02.10.10.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-6 - 2018-07-17 23.50.06.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-7 - 2018-07-17 21.25.57.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-8 - 2018-07-17 19.05.58.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-9_2 - 2018-07-17 17.18.45.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-10.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-11.ndpis',
	'/media/tpisano/FAT32/nanozoomer/M9332_Rabies_2-12 - 2018-07-18 19.53.50.ndpis']


    #location to save data
    dst = 	'/media/tpisano/FAT32/nanozoomer/output_level2'

    #elastix parameter file to use for alignment - fast is typically sufficient, but for better results use 'align_slices_elastix_parameters.txt'
    #parameters = ['/media/tpisano/FAT32/pyalign/elastix_files/align_slices_elastix_parameters_full.txt']
    parameters = ['/media/tpisano/FAT32/pyalign/elastix_files/align_slices_elastix_parameters_fullscale.txt']
    #parameters = ['/media/tpisano/FAT32/pyalign/elastix_files/align_slices_elastix_parameters_fast.txt']

    #sectioning depth
    section_depth = 40 #um

    #nanozoomer level of resolution to use; 1 = full resolution - typically this is too large for most things including fiji, 6 = pretty small
    level = 2

    #channel to use for alignment e.g. 'Trtc', 'Brighfield'
    channel= 'Trtc'
    verbose=True

    #collect and build structure
    df = read_ndpis(src)

    #output
    makedir(dst)
    dst0 = os.path.join(dst, 'data'); makedir(dst0)
    pnts_dst = outfld
    
    #now load and collect sections and save them out
    find_pixels_from_contours(pnts_dst, dst0, df, channel)
    
    #need to expand images to the largest section
    nsrc = os.path.join(dst0, channel)
    #other_channel_folders = [xx for xx in listdirfull(dst0) if dst1 not in xx and '_registration' not in xx and '_aligned' not in xx and nsrc not in xx]
    pad_first_image(nsrc)
    
    #now register
    align_sections(nsrc, dst0, parameters, other_channel_folders=False, clean=True)

    