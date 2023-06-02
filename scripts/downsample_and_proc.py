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
import scipy
from preprocess_aux import *


def main(fn,pleth_chan,dia_chan,ekg_chan,temp_chan,v_in,save_path):

    if save_path is None:
        save_path = os.path.split(fn)[0]

    mmap,meta = load_mmap(fn)
    raw_dia,sr_dia = load_dia_emg(mmap,meta,dia_chan)
    dia_df,dia_sub,sr_dia_sub,HR,dia_filt = filt_int_ds_dia(raw_dia,sr_dia)
    if ekg_chan !=-1:
        HR = extract_hr_channel(mmap,meta,ekg_chan)

    if pleth_chan<0:
        pleth = []
        sr_pleth = sr_dia_sub
    else:
        # TODO: correct flowmeter/pdiff calculations
        pleth,sr_pleth = load_ds_pdiff(mmap,meta,pleth_chan)
        # pleth = pleth/np.std(pleth)
        print('Using flowmeter calibrations')
        pleth = data.calibrate_flowmeter(pleth,vin=v_in)

    if temp_chan>0:
        temperature = extract_temp(mmap,meta,temp_chan)
    else:
        temperature = []



    #Make sure the subsampled dia and pleth have identical SR
    assert(sr_pleth == sr_dia_sub)
    t = np.arange(0, len(dia_sub)/sr_pleth, 1 / sr_pleth)
    t = t[:len(dia_sub)]

    # Map heart ratea into t
    hr_idx = np.searchsorted(t,HR['t'])
    new_hr = pd.DataFrame()
    new_hr['t'] = t
    new_hr['hr'] = np.nan
    new_hr.iloc[hr_idx,-1] = HR['hr (bpm)'].values
    new_hr.interpolate(limit_direction='both',inplace=True)


    # Save the downsampled data to a mat file
    data_dict = {
        'pleth':pleth,
        'dia':dia_sub,
        'sr':sr_pleth,
        'hr_bpm':new_hr['hr'].values,
        'temperature':temperature,
        't':t
    }
    save_fn,prefix = make_save_fn(fn,save_path)
    sio.savemat(save_fn,data_dict,oned_as='column')


    # Save the extracted diaphragm to a csv
    # But strip the data referenced to the 10K sampling
    dia_df.drop(['on_samp','off_samp','duration_samp','pk_samp'],axis=1,inplace=True)
    dia_df.to_csv(os.path.join(save_path,f'{prefix}_dia_stat.csv'))

    dia_df['on_sec'].to_csv(os.path.join(save_path,f'{prefix}_dia_onsets.csv'),index=False)

    data_dict_raw = {
        'dia':dia_filt,
        't':np.arange(0,len(dia_filt)/sr_dia,1/sr_dia)
    }
    save_fn,prefix = make_save_fn(fn,save_path,save_name='_filtered_dia')
    sio.savemat(save_fn,data_dict_raw,oned_as='column')

#TODO: Use a different function for processiong PDIFF vs Flowmeter
@click.command()
@click.argument('fn')
@click.option('-p','--pleth_chan','pleth_chan',default=5)
@click.option('-d','--dia_chan','dia_chan',default=0)
@click.option('-e','--ekg_chan','ekg_chan',default=2)
@click.option('-t','--temp_chan','temp_chan',default=7)
@click.option('-v','--v_in','v_in',default=9,type=float)
def batch(fn,pleth_chan,dia_chan,v_in,ekg_chan,temp_chan):
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
                    main(fname,pleth_chan,dia_chan,ekg_chan,temp_chan,v_in,root)
                    if pleth_chan>=0:
                        matlab_cmd_string = "matlab -nosplash -nodesktop -nojvm -r bm_mat_proc('" + fname + "')"
                        os.system(matlab_cmd_string)
                    else:
                        print('No pleth signal so not performing BM')

                    # except:
                    #     print('='*50)
                    #     print(f'Failure on file {fname}')
                    #     print('='*50)
    else:
        root = os.path.split(fn)[0]
        main(fn, pleth_chan, dia_chan, ekg_chan,temp_chan,v_on,root)
        if pleth_chan>=0:
            matlab_cmd_string = "matlab -nosplash -nodesktop -nojvm -r bm_mat_proc('" + fn + "')"
            os.system(matlab_cmd_string)
        else:
            print('No pleth signal so not performing BM')


if __name__=='__main__':
    batch()
