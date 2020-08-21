"""Routines for data import and manipulation."""
import sys
import numpy as np
import readSGLX
from pathlib import Path
sys.path.append('../')
from utils.ephys.signal import binary_onsets

def get_tvec(x_sync,sync_timestamps,sr):
    '''
    Get a time vector for NI that uses the sync signal

    :param x_sync:
    :param sync_timestamps:
    :return: tvec
    '''
    tvec = np.empty_like(x_sync)
    onsets,offsets = binary_onsets(x_sync,1)
    n_ons = len(onsets)
    n_ts = len(sync_timestamps)
    if n_ons!=n_ts:
        raise ValueError(f"number of detected onsets {n_ons} does not match number expected {n_ts}")

    # Map all detected onsets to a timestamp and linearly interpolate
    for ii in range(len(onsets)-1):
        nsamps = onsets[ii+1]-onsets[ii]
        temp = np.linspace(sync_timestamps[ii],sync_timestamps[ii+1],nsamps)
        tvec[onsets[ii]:onsets[ii+1]] = temp

    # get times before first synch signal
    first_t = sync_timestamps[0]-onsets[0]/sr
    first_seg = np.linspace(first_t,sync_timestamps[0],onsets[0])
    tvec[:onsets[0]] = first_seg

    # get times after last synch signal
    last_seg_length = len(x_sync)-onsets[-1]
    last_seg = np.linspace(sync_timestamps[-1],sync_timestamps[-1] + last_seg_length/sr,last_seg_length)
    tvec[onsets[-1]:] = last_seg

    return(tvec)


def get_ni_analog(ni_bin_fn, chan_id):
    '''
    Convinience function to load in a NI analog channel
    :param ni_bin_fn: filename to load from
    :param chan_id: channel index to load
    :return: analog_dat
    '''
    meta = readSGLX.readMeta(Path(ni_bin_fn))
    bitvolts = readSGLX.Int2Volts(meta)
    ni_dat = readSGLX.makeMemMapRaw(ni_bin_fn,meta)
    analog_dat = ni_dat[chan_id]*bitvolts

    return(analog_dat)

