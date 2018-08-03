#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:15:33 2018

@author: tpisano
"""
import os, numpy as np, sys, subprocess as sp
from skimage.external import tifffile
from tools.registration.register import elastix_command_line_call
#really important to have order 1 interpolation of rigid
def listdirfull(src, keyword=False):
    if not keyword: fls = [os.path.join(src, xx) for xx in os.listdir(src) if 'Thumbs.db' not in xx]
    if keyword: fls = [os.path.join(src, xx) for xx in os.listdir(src) if 'Thumbs.db' not in xx and keyword in xx]
    fls.sort()
    return fls

def makedir(src):
    if not os.path.exists(src): os.mkdir(src)
    return
 
def elastix_command_line_call(fx, mv, out, parameters, verbose=True):
    '''Wrapper Function to call elastix using the commandline, this can be time consuming

    Inputs
    -------------------
    fx = fixed path (usually Atlas for 'normal' noninverse transforms)
    mv = moving path (usually volume to register for 'normal' noninverse transforms)
    out = folder to save file
    parameters = list of paths to parameter files IN ORDER THEY SHOULD BE APPLIED

    Outputs
    --------------
    ElastixResultFile = '.tif' or '.mhd' result file
    TransformParameterFile = file storing transform parameters

    '''
    e_params=['elastix', '-f', fx, '-m', mv, '-out', out]

    ###adding elastix parameter files to command line call
    for x in range(len(parameters)):
        e_params.append('-p')
        e_params.append(parameters[x])
    if verbose: sys.stdout.write('Elastix Command:\n{}\n...'.format(e_params)); sys.stdout.flush()

    #set paths
    TransformParameterFile = os.path.join(out, 'TransformParameters.{}.txt'.format((len(parameters)-1)))
    ElastixResultFile = os.path.join(out, 'result.{}.tif'.format((len(parameters)-1)))

    #run elastix:
    try:
        if verbose: print ('Running Elastix, this can take some time....\n')
        sp.call(e_params)
        if verbose:
            sys.stdout.write('Past Elastix Commandline Call'); sys.stdout.flush()
    except RuntimeError, e:
        sys.stdout.write('\n***RUNTIME ERROR***: {} Elastix has failed. Most likely the two images are too dissimiliar.\n'.format(e.message)); sys.stdout.flush()
        pass
    if os.path.exists(ElastixResultFile) == True:
        if verbose: sys.stdout.write('Elastix Registration Successfully Completed\n'); sys.stdout.flush()
    #check to see if it was MHD instead
    elif os.path.exists(os.path.join(out, 'result.{}.mhd'.format((len(parameters)-1)))) == True:
        ElastixResultFile = os.path.join(out, 'result.{}.mhd'.format((len(parameters)-1)))
        if verbose: sys.stdout.write('Elastix Registration Successfully Completed\n'); sys.stdout.flush()
    else:
        sys.stdout.write('\n***ERROR***Cannot find elastix result file\n: {}'.format(ElastixResultFile)); sys.stdout.flush()
        return

    return ElastixResultFile, TransformParameterFile  

#%%
if __name__ == '__main__':
    
    #elastix parameter file to use for alignment typically both Order1_Par0000affine.txt follwed by Order2_Par0000bspline.txt
    parameters = ['/media/tpisano/FAT32/volumetric/Order1_Par0000affine.txt',
                  #'/media/tpisano/FAT32/volumetric/Order2_Par0000bspline.txt'
                  ]
    #Path to Atlas, note that this must be oriented in the same way as your dataset, see FIJI/ImageJ for ways to reorient
    atlas_path =  '/media/tpisano/FAT32/volumetric/average_template_25_coronal.tif'
    
    #location of aligned files from serial_section_processing.py
    src = '/media/tpisano/FAT32/fast/Trtc_aligned'
    src = '/media/tpisano/FAT32/nanozoomer/output_level2/data/Trtc_aligned'
    src = '/media/tpisano/FAT32/nanozoomer/output/data/Trtc_aligned'
    
    #location to save elastix input into
    out = '/media/tpisano/FAT32/volumetric/elastix'
    
    #######DO NOT NEED TO TOUCH BELOW
    makedir(out)
    #load src, concatenate and make sure appropriate bitdepth
    vol = np.asarray([tifffile.imread(xx).astype('uint8') for xx in listdirfull(src)]) #need to do the uint8 thing
    nsrc = src+'.tif'
    tifffile.imsave(nsrc, vol.astype('uint16')) #needs to be 16 bit!

    #now run elastix:
    elastix_command_line_call(fx = atlas_path, mv = nsrc, out = out, parameters=parameters)
    
if False:
    #elastix parameter file to use for alignment typically both Order1_Par0000affine.txt follwed by Order2_Par0000bspline.txt
    parameters = ['/media/tpisano/FAT32/volumetric/elastix_bsplineopt/Order2_Par0000bspline.txt']
    
    #Path to Atlas, note that this must be oriented in the same way as your dataset, see FIJI/ImageJ for ways to reorient
    atlas_path =  '/media/tpisano/FAT32/volumetric/average_template_25_coronal.tif'
    
    #location of aligned files from serial_section_processing.py
    src = '/media/tpisano/FAT32/volumetric/elastix_bsplineopt/affine.tif'
    
    #location to save elastix input into
    out =  '/media/tpisano/FAT32/volumetric/elastix_bsplineopt'
    
    #now run elastix:
    elastix_command_line_call(fx = atlas_path, mv = src, out = out, parameters=parameters)