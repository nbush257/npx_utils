'''
The recorded auxiliary data (e.g. diaphragm and pleth) is sampled at 10K.
While the high sampling rate is critical to acquire good EMG, it is excessive for both
the integrated and the pleth.

This script does the following:
1) Downsamples the plethysmography by 10x to be input into breathmetrics (BM does filtering and processing)
2) Filters the EMG (300-5K)
3) Integrates the EMG with a ______ exponential window
4) Downsamples the integrated EMG by 10x to match the pleth
5) Extracts features from the integrated EMG.
6) Saves a .mat file with the downsampled integrated EMG and the Pleth
7) Saves a .csv with the diaphragm features
'''
import readSGLX
import numpy as np
import scipy.signal as sig
import scipy.io.matlab as sio
import pandas as pd
import data
import proc
from ephys.signal import remove_EKG
from pathlib import Path

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
    sr = readSGLX.Int2Volts(meta)
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
    return(dat,sr)


def filt_int_ds_dia(x,sr,ds_factor=10):
    assert(type(ds_factor) is int )
    xx = x-np.mean(x)
    xx = remove_EKG(xx,sr)
    xx[np.isnan(xx)] = 0

    # Filter
    sos = sig.butter(4 ,[300/sr/2,5000/sr/2],btype='bandpass',output='sos')
    xf = sig.sosfilt(sos,xx)

    #integrate

    filt = pd.DataFrame(np.abs(xx))









def main():
    pass



