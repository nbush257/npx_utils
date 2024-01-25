#TODO: Update docstring for this script.
'''
The recorded auxiliary data (e.g. diaphragm and pleth) is sampled at 10K.
While the high sampling rate is critical to acquire good EMG, it is excessive for both
the integrated and the pleth.

This script does the following:
1) Downsamples the Flow/Pdiff  traces by 10x to be input into breathmetrics (BM does filtering and processing)
2) Filters the EMG (300-5K)
3) Integrates the EMG with a triangular window
4) Downsamples the integrated EMG by 10x to match the pleth
5) Extracts features from the integrated EMG.
6) Extracts heart rate from EKG channel if recorded
6) Saves a .mat file with the downsampled integrated EMG,flow/pdiff,heart rrate, heart beats
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
from scipy.ndimage import median_filter
import pandas as pd
import scipy
from preprocess_aux import *

#TODO: test crop
def crop_traces(t,x):
    '''
    Utility to crop both the time trace and an arbitrary vector to the same length(shortest lenght)
    useful to preven off by one errors
    :param t:
    :param x:
    :return:
    '''
    tlen = np.min([len(x), len(t)])
    return(t[:tlen],x[:tlen])


def main(fn, pdiff_chan, flowmeter_chan, dia_chan, ekg_chan, temp_chan, v_in, inhale_pos, save_path):
    DS_FACTOR=10
    # INIT output variables
    pdiff = []
    flow = []
    dia_filt= []
    dia_sub = []
    temperature = []
    heartbeats = []
    hr_bpm = []

    if inhale_pos:
        inhale_dir=1
    else:
        inhale_dir=-1


    if save_path is None:
        save_path = os.path.split(fn)[0]

    # LOAD Memory map from SGLX
    mmap,meta = load_mmap(fn)
    sr = readSGLX.SampRate(meta)
    # Get tvec
    t = get_tvec_from_mmap(mmap,meta)
    t = t[::DS_FACTOR]
    sr_sub = sr/DS_FACTOR

    print(f'Sampling rate is {sr}')
    print(f'Downsampling to {sr_sub}')

    # Process diaphragm
    # Must do before explicit ekg processing because it attempts
    # to find the heartbeats, but it is not as good as the
    # explicit EKG channel and so we want to overwrite heartbeats with those data if they exist
    if dia_chan>=0:
        print('Processing diaphragm')
        raw_dia,sr_dia = load_dia_emg(mmap,meta,dia_chan)
        dia_df,dia_sub,sr_dia_sub,HR,dia_filt,heartbeats = filt_int_ds_dia(raw_dia,sr_dia,ds_factor=DS_FACTOR)
        t,dia_sub = crop_traces(t,dia_sub)

    # Process EKG
    if ekg_chan >=0:
        print('Processing EKG')
        HR,heartbeats = extract_hr_channel(mmap,meta,ekg_chan)

    # Process PDIFF
    if pdiff_chan>=0:
        print('Processing pressure diffrential sensor')
        pdiff,sr_pdiff = load_ds_pdiff(mmap, meta, pdiff_chan,ds_factor=DS_FACTOR,inhale_dir=inhale_dir)
        t,pdiff = crop_traces(t,pdiff)

    # Process Flowmeter
    if flowmeter_chan>=0:
        print('Processing flowmeter')
        flow,sr_flow = load_ds_process_flowmeter(mmap,meta,flowmeter_chan,v_in,ds_factor=DS_FACTOR,inhale_dir=inhale_dir)
        t,flow = crop_traces(t,flow)

    # Process Temperature
    if temp_chan!=-1:
        print('Processing temperature')
        temperature = extract_temp(mmap,meta,temp_chan,ds_factor=DS_FACTOR)
        t,temperature = crop_traces(t,temperature)

    # Map heart rate into t
    if dia_chan>=0 or ekg_chan>=0:
        hr_idx = np.searchsorted(t,HR['t'])
        new_hr = pd.DataFrame()
        new_hr['t'] = t
        new_hr['hr'] = np.nan
        new_hr.iloc[hr_idx,-1] = HR['hr (bpm)'].values
        new_hr.interpolate(limit_direction='both',inplace=True)
        hr_bpm = new_hr['hr'].values

    # Save the downsampled data to a mat file
    data_dict = {
        'pdiff':pdiff,
        'flowmeter':flow,
        'dia':dia_sub,
        'sr':sr_sub,
        'hr_bpm':hr_bpm,
        'heartbeats':heartbeats,
        'temperature':temperature,
        't':t
    }
    save_fn,prefix = make_save_fn(fn,save_path)
    sio.savemat(save_fn,data_dict,oned_as='column')


    if dia_chan>=0:
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

@click.command()
@click.argument('fn')
@click.option('-p','--pdiff_chan','pdiff_chan',default=4,show_default=True)
@click.option('-f','--flowmeter_chan','flowmeter_chan',default=5,show_default=True)
@click.option('-d','--dia_chan','dia_chan',default=0,show_default=True)
@click.option('-e','--ekg_chan','ekg_chan',default=2,show_default=True)
@click.option('-t','--temp_chan','temp_chan',default=7,show_default=True)
@click.option('-v','--v_in','v_in',default=9,type=float,show_default=True)
@click.option('-i','--inhale_pos',is_flag=True,default=False,show_default=True)
@click.option('-s','--save_path','save_path',default=None,show_default=True)
def batch(fn, pdiff_chan, flowmeter_chan, dia_chan, ekg_chan, temp_chan, v_in, inhale_pos, save_path):
    '''
    Set chan to -1 if no data is recorded.
    '''
    if os.path.isdir(fn):
        print('Running as batch\n')
        for root,dirs,files in os.walk(fn):
            r = re.compile('.*nid.*bin')
            flist = list(filter(r.match, files))
            if len(flist)>0:
                print('Processing files:')
                print(flist)
                for ff in flist:
                    fname = os.path.join(root,ff)
                    print(fname)
                    # try:
                    main(fname,pdiff_chan,flowmeter_chan,dia_chan,ekg_chan,temp_chan,v_in,inhale_pos,root)
                    if pdiff_chan>=0 or flowmeter_chan>=0:
                        matlab_cmd_string = "matlab -batch bm_mat_proc('" + fname + "')"
                        os.system(matlab_cmd_string)
                    else:
                        print('No airflow signal so not performing BM')
    else:
        print('Running as one')
        root = os.path.split(fn)[0]
        main(fn, pdiff_chan, flowmeter_chan, dia_chan, ekg_chan, temp_chan, v_in, inhale_pos, root)
        if pdiff_chan>=0 or flowmeter_chan>=0:
            matlab_cmd_string = "matlab -batch bm_mat_proc('" + fn + "')"
            os.system(matlab_cmd_string)
        else:
            print('No pdiff signal so not performing BM')


if __name__=='__main__':
    batch()
