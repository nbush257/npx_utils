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
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../utils')
import readSGLX
import numpy as np
import scipy.signal as sig
import scipy.io.matlab as sio
import pandas as pd
import data
import proc
from ephys.signal import remove_EKG
from pathlib import Path
import click

def load_mmap(fn):
    '''
    Load a memory map of the auxiliary channels
    :param fn: filename to nidaq .bin file
    :return:
            mmap - memory map to the aux data
            sr - sampling rate of aux data
    '''
    meta = readSGLX.readMeta(Path(fn))
    mmap = readSGLX.makeMemMapRaw(fn,meta)

    return(mmap,meta)


def load_ds_pleth(mmap,meta,chan_id,ds_factor=10):
    '''
    Load and downsample the pleth data
    :param mmap: mmap
    :param meta: metadata dict
    :param chan_id: Pleth channel index
    :param ds_factor: downsample factor
    :return:
            dat- downsampled pleth data
            sr_sub - new sampling rate
    '''
    assert(type(ds_factor) is int )
    bitvolts = readSGLX.Int2Volts(meta)
    sr = readSGLX.SampRate(meta)
    dat = mmap[chan_id,::ds_factor]*bitvolts
    sr_sub = sr/ds_factor

    return(dat,sr_sub)


def load_dia_emg(mmap,meta,chan_id):
    '''
    Read the raw diaphragm emg
    :param mmap: memory ampped aux data
    :param meta: meta dict
    :param chan_id: channel index of the diaphragm
    :return:
        dat - the raw diaphramg
        sr - the smapling rate of the diaphragm recording
    '''
    bitvolts = readSGLX.Int2Volts(meta)
    sr = readSGLX.SampRate(meta)
    dat = mmap[chan_id]*bitvolts
    dat = dat-np.mean(dat)
    return(dat,sr)


def filt_int_ds_dia(x,sr,ds_factor=10):
    assert(type(ds_factor) is int )
    dia_df,phys_df,dia_int = proc.proc_dia(x,sr,win=0.25)

    dia_sub = dia_int[::ds_factor]
    sr_sub = sr/ds_factor

    return(dia_df,dia_sub,sr_sub)


def make_save_fn(fn,save_path,save_name='_aux_downsamp'):

    load_path,no_path = os.path.split(fn)
    prefix = no_path.replace('.nidq.bin','')
    save_fn = os.path.join(save_path,prefix+save_name+'.mat')
    return(save_fn)


@click.command()
@click.argument('fn')
@click.option('-p','--pleth_chan','pleth_chan',default=0)
@click.option('-d','--dia_chan','dia_chan',default=1)
@click.option('-s','--save_path','save_path',default=None)
def main(fn,pleth_chan,dia_chan,save_path):

    if save_path is None:
        save_path = os.path.split(fn)[0]


    mmap,meta = load_mmap(fn)
    pleth,sr_pleth = load_ds_pleth(mmap,meta,pleth_chan)
    raw_dia,sr_dia = load_dia_emg(mmap,meta,dia_chan)
    dia_df,dia_sub,sr_dia_sub = filt_int_ds_dia(raw_dia,sr_dia)



    #Make sure the subsampled dia and pleth have identical SR
    assert(sr_pleth == sr_dia_sub)

    # Save the downsampled data to a mat file
    data_dict = {
        'pleth':pleth,
        'dia':dia_sub,
        'sr':sr_pleth
    }
    save_fn = make_save_fn(fn,save_path)
    sio.savemat(save_fn,data_dict)

    # Save the extracted diaphragm to a csv
    # But strip the data referenced to the 10K sampling
    dia_df.drop(['on_samp','off_samp','duration_samp','pk_samp'],axis=1,inplace=True)
    dia_df.to_csv(os.path.join(save_path,'dia_stat.csv'))

    dia_df['on_sec'].to_csv(os.path.join(save_path,'dia_onsets.csv'),index=False)

if __name__=='__main__':
    main()
