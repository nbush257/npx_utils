import scipy.integrate

import readSGLX
import numpy as np
import scipy.signal as sig
import scipy.io.matlab as sio
import proc,data
from pathlib import Path
from sklearn.mixture import BayesianGaussianMixture
from scipy.ndimage import median_filter
import pandas as pd
import warnings
import os

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


def get_tvec(dat,sr):
    tvec = np.linspace(0,len(dat)/sr,len(dat))
    return(tvec)


def get_tvec_from_fn(fn):
    '''
    Overload get tvec to work on a nidaq filename
    :param fn: Path object to a Nidaq file
    :return: tvec
    '''
    mmap,meta = load_mmap(fn)
    tvec = get_tvec_from_mmap(mmap,meta)
    return(tvec)


def get_tvec_from_mmap(mmap,meta):
    '''
    Extract the timevector given a memory map and a metafile
    :param mmap: A nidaq memory map object of aux data
    :param meta: Meta data as extracted from readSGLX
    :return: tvec
    '''
    sr = readSGLX.SampRate(meta)
    n_samps = mmap.shape[1]
    tvec = np.linspace(0,n_samps/sr,n_samps)
    return(tvec)


def load_ds_pdiff(mmap,meta,chan_id,ds_factor=10,winsize=5,inhale_dir=-1):
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

    assert(type(ds_factor) is int)
    bitvolts = readSGLX.Int2Volts(meta)
    sr = readSGLX.SampRate(meta)
    sr_sub = sr / ds_factor
    dat = mmap[chan_id,::ds_factor]*bitvolts
    dat = dat*inhale_dir

    # Do not do any baseline correction on the PDIFF because it is AC.
    return(dat,sr_sub)


def load_ds_process_flowmeter(mmap,meta,chan_id,vin=9,ds_factor=10,inhale_dir=-1):
    assert (type(ds_factor) is int)
    bitvolts = readSGLX.Int2Volts(meta)
    sr = readSGLX.SampRate(meta)
    flow = mmap[chan_id, ::ds_factor] * bitvolts
    sr_sub = sr / ds_factor
    winsize = 5
    # Calibrate voltage to flow
    flow_calibrated = data.calibrate_flowmeter(flow, vin=vin)
    # Correct for bias flow
    flow_calibrated_corrected = data.baseline_correct_integral(flow_calibrated,sr=sr,winsize=winsize)
    # Make inhalation updward deflections
    flow_calibrated_corrected = flow_calibrated_corrected * inhale_dir
    return(flow_calibrated_corrected,sr_sub)


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


def filt_int_ds_dia(x,sr,ds_factor=10,rel_height=0.95):
    '''
    Filter, integrate and downsample the diaphragm. Detect and summarize the diaphragm bursts
    Uses medfilt to smooth so it is a little slow, but it is worth it.
    :param x:
    :param sr:
    :param ds_factor:
    :return:
    '''
    assert(type(ds_factor) is int)

    #Remove the EKG artifact
    print('Removing the EKG...')
    dia_filt,pulse = remove_EKG(x,sr,thresh=2)
    dia_filt[np.isnan(dia_filt)] = np.nanmedian(dia_filt)


    # Filter for high frequency signal

    sos = sig.butter(2,[300/sr/2,5000/sr/2],btype='bandpass',output='sos')
    dia_filt = sig.sosfilt(sos,dia_filt)

    # Use medfilt to get the smoothed rectified EMG
    print('Smoothing the rectified trace...')

    window_length = int(0.05*np.round(sr))+1
    if window_length%2==0:
        window_length+=1
    dd = median_filter(np.abs(dia_filt),window_length)
    # Smooth it out a little more
    window_length = int(0.01*np.round(sr))+1
    if window_length%2==0:
        window_length+=1
    dia_smooth = sig.savgol_filter(dd,window_length=window_length,polyorder=1)

    # Downsample because we don't need this at the original smapling rate
    dia_sub = dia_smooth[::ds_factor]
    sr_sub = sr/ds_factor

    # get the burst statistics
    warnings.filterwarnings('ignore')
    dia_df = proc.burst_stats_dia(dia_sub,sr_sub,rel_height=rel_height)
    warnings.filterwarnings('default')

    HR,heartbeats = get_hr_from_dia(pulse/ds_factor,dia_df,sr_sub)

    # Normalize the integrated diaphragm to a z-score.
    dia_df['amp_z'] = dia_df['amp']/np.std(dia_sub)
    dia_sub = dia_sub/np.std(dia_sub)
    print('Done processing diaphragm')

    return(dia_df,dia_sub,sr_sub,HR,dia_filt,heartbeats)


def remove_EKG(x,sr,thresh=2):
    '''
    :param x: diaphragm (or other) emg
    :param sr: sample rate
    :return: y ekg filtered out
            pks - pulse times
    '''
    warnings.filterwarnings('ignore')
    sos = sig.butter(2,[5/sr/2,500/sr/2],btype='bandpass',output='sos')
    xs = sig.sosfilt(sos,x)
    pks = sig.find_peaks(xs,prominence=thresh*np.std(xs),distance=0.05*sr)[0]
    pks = pks[1:-1]
    amps = xs[pks]
    win = int(0.010 *sr)
    y = x.copy()
    ekg = np.zeros([2*win,len(pks)])
    for ii,pk in enumerate(pks):
        try:
            ekg[:,ii] = x[pk-win:pk+win]
        except:
            pass

    ekg_std = np.std(ekg[:30],0) +np.std(ekg[-30:],0)
    ekg_std = np.log(ekg_std)
    mask = np.logical_not(np.isfinite(ekg_std))
    ekg_std[mask] = np.nanmedian(ekg_std)


    bgm = BayesianGaussianMixture(n_components=2)
    cls = bgm.fit_predict(ekg_std[:,np.newaxis])
    cls[cls==0]=-1
    m0 = np.nanmean(ekg_std[cls==-1])
    m1 = np.nanmean(ekg_std[cls==1])
    if m0>m1:
        cls = -cls

    ww = int(.0005 * sr)
    ww += ww % 2 - 1

    for ii,pk in enumerate(pks):
        if pk-win<0:
            continue
        if (pk+win)>len(y):
            continue
        if cls[ii]==-1:
            sm_ekg = sig.savgol_filter(ekg[:,ii],ww,1)
            y[pk - win:pk + win] -= sm_ekg
        else:
            med_ekg = np.nanmedian(ekg[:,ii-5:ii+5],1)
            med_amp = np.median(amps[ii-5:ii+5])
            scl = amps[ii]/med_amp
            y[pk - win:pk + win] -=med_ekg*scl

    y[np.isnan(y)] = np.nanmedian(y)
    warnings.filterwarnings('default')
    return(y,pks)


def get_hr_from_dia(pks,dia_df,sr):
    'computes the avergae heart rate from the diaphragm - not as good as getting it from a dedicated channel'
    ons = dia_df['on_samp']
    offs = dia_df['off_samp']
    for on,off in zip(ons,offs):
        mask = np.logical_not(
            np.logical_and(
                pks>on,
                pks<off
            )
        )
        pks = pks[mask]

    pulse = pd.DataFrame()
    pulse['hr (bpm)'] = 60*sr/np.diff(pks)
    hr_smooth = pulse.rolling(50,center=True).median()
    hr_smooth.interpolate(limit_direction='both',inplace=True)
    hr_smooth['t']=pks[:-1]/sr
    return(hr_smooth,pks/sr)


def extract_hr_channel(mmap,meta,ekg_chan=2):
    '''
    If the ekg is recorded on a separate channel, extract it here
    return: bpm, pulse times
    '''
    bitvolts = readSGLX.Int2Volts(meta)
    sr = readSGLX.SampRate(meta)
    dat = mmap[ekg_chan]*bitvolts
    dat = dat-np.mean(dat)

    sos = sig.butter(8,[100/sr/2,1000/sr/2],btype='bandpass',output='sos')
    xs = sig.sosfilt(sos,dat)
    pks = sig.find_peaks(xs,prominence=5*np.std(xs),distance=int(0.05*sr))[0]

    pulse = pd.DataFrame()
    pulse['hr (bpm)'] = 60*sr/np.diff(pks)
    bpm = pulse.rolling(50,center=True).median()
    bpm.interpolate(limit_direction='both',inplace=True)
    bpm['t']=pks[:-1]/sr
    return(bpm,pks/sr)


def extract_temp(mmap,meta,temp_chan=7,ds_factor=10):
    """
    Extract the temperature from the FHC DC temp controller. Assumes the manufacturers calibration
    :param mmap:
    :param meta:
    :param temp_chan:
    :return:
    """
    assert(type(ds_factor) is int)
    bitvolts = readSGLX.Int2Volts(meta)
    sr = readSGLX.SampRate(meta)
    dat = mmap[temp_chan]*bitvolts
    # 0v=25C, 2V = 45C, 100mv=1C
    vout_map = [0,2]
    temp_map = [25,45]
    temp_f = scipy.interpolate.interp1d(vout_map, temp_map)
    temp_out = temp_f(dat)
    temp_out = scipy.signal.savgol_filter(temp_out,101,1)[::ds_factor]
    return(temp_out)


def make_save_fn(fn,save_path,save_name='_aux_downsamp'):

    load_path,no_path = os.path.split(fn)
    prefix = no_path.replace('.nidq.bin','')
    save_fn = os.path.join(save_path,prefix+save_name+'.mat')
    return(save_fn,prefix)

