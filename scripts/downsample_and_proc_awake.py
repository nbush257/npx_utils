#TODO: Update docstring for this script.
'''
The recorded auxiliary data (e.g. diaphragm and pleth) is sampled at 10K.
While the high sampling rate is critical to acquire good EMG, it is excessive for both
the integrated and the pleth.

This script does the following:
1) Downsamples the plethysmography by 10x to be input into breathmetrics (BM does filtering and processing)
2) Filters the EMG (300-5K)
3) Integrates the EMG with a triangular window
4) Downsamples the integrated EMG by 10x to match the pleth
5) Extracts features from the integrated EMG.
6) Saves a .mat file with the downsampled integrated EMG and the Pleth
7) Saves a .csv with the diaphragm features
'''
import os
import re
import sys
import warnings
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../utility')
import readSGLX
import numpy as np
import scipy.signal as sig
import scipy.io.matlab as sio
import proc,data
from pathlib import Path
import click
from sklearn.mixture import BayesianGaussianMixture
from scipy.ndimage.filters import median_filter
import pandas as pd

#TODO: Use a different function for processiong PDIFF vs Flowmeter
#TODO: Call only one bm_mat_proc file across the awake/anest scripts
def main(fn,pleth_chan,v_in,save_path):

    if save_path is None:
        save_path = os.path.split(fn)[0]

    mmap,meta = load_mmap(fn)

    pleth,sr_pleth = load_ds_pleth(mmap,meta,pleth_chan)
    # pleth = pleth/np.std(pleth)
    print('Using flowmeter calibrations')
    pleth = data.calibrate_flowmeter(pleth,vin=v_in)


    #Make sure the subsampled dia and pleth have identical SR
    t = np.arange(0, len(pleth)/sr_pleth, 1 / sr_pleth)
    t = t[:len(pleth)]


    # Save the downsampled data to a mat file
    data_dict = {
        'pleth':pleth,
        'sr':sr_pleth,
        't':t
    }
    save_fn,prefix = make_save_fn(fn,save_path)
    sio.savemat(save_fn,data_dict,oned_as='column')


@click.command()
@click.argument('fn')
@click.option('-p','--pleth_chan','pleth_chan',default=0)
@click.option('-v','--v_in','v_in',default=9,type=float)
@click.option('-s','--save_path','save_path',default=None)
def batch(fn,pleth_chan,v_in):
    '''
    Set pleth chan to -1 if no pleth is recorded.
    :param fn:
    :param pleth_chan:
    :param dia_chan:
    :param save_path:
    :return:
    '''
    if os.path.isdir(fn):
        print('Running as batch\n')
        for root,dirs,files in os.walk(fn):
            r = re.compile('.*nid.*bin')
            flist = list(filter(r.match, files))
            if len(flist)>0:
                print(flist)
                for ff in flist:
                    fname = os.path.join(root,ff)
                    print(fname)
                    # try:
                    main(fname,pleth_chan,v_in,root)
                    if pleth_chan>=0:
                        matlab_cmd_string = "matlab -nosplash -nodesktop -nojvm -r bm_mat_proc_awake('" + fname + "')"
                        os.system(matlab_cmd_string)
                    else:
                        print('No pleth signal so not performing BM')

    else:
        root = os.path.split(fn)[0]
        main(fn, pleth_chan,v_in,root)
        if pleth_chan>=0:
            matlab_cmd_string = "matlab -nosplash -nodesktop -nojvm -r bm_mat_proc_awake('" + fn + "')"
            os.system(matlab_cmd_string)
        else:
            print('No pleth signal so not performing BM')

if __name__=='__main__':
    batch()
