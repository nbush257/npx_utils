import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import scipy
import scipy.signal as sig
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import FastICA as ICA


def bwfilt(x,fs,low=300,high=10000):
    b,a = scipy.signal.butter(2,[low/fs/2,high/fs/2],btype='bandpass')
    y = scipy.signal.filtfilt(b,a,x)
    return(y)


def find_nearest(array,value):
    '''
    Find the index of the nearest value in "array" to the value of "value"
    Useful for matching indices of timestamps from sources with different sampling rates
    Parameters
    ----------
    array : an array of values to be mapped into
    value : a value to map into "array"

    Returns
    -------
    idx - index in array to which "value" is nearest
    '''

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def integrator(x,sr,span=0.5,acausal=True):
    '''

    :param x: raw
    :param sr: sampling rate
    :param span: in seconds
    :param acausal: if true, run forward and backward
    :return:
    '''

    span*=sr# convert to samples
    span = int(span)
    if acausal:
        df = pd.DataFrame()
        df['f'] = np.abs(x)
        df['b'] = np.abs(x[::-1])
        aa = df.ewm(span=span).mean()
        temp = np.vstack([aa['f'],aa['b'][::-1]]).T
        integrated = np.mean(temp,axis=1)
    else:
        df = pd.DataFrame()
        df['f'] = np.abs(x)
        integrated = df.ewm(span=span).mean().values.ravel()
    return(integrated)


def binary_onsets(x,thresh):
    '''
    Get the onset and offset samples of a binary signal (
    :param x: signal
    :param thresh: Threshold
    :return: ons,offs
    '''
    xbool = x>thresh

    ons = np.where(np.diff(xbool.astype('int'))==1)[0]
    offs = np.where(np.diff(xbool.astype('int'))==-1)[0]
    if xbool[0]:
        offs = offs[1:]
    if xbool[-1]:
        ons = ons[:-1]
    if len(ons)!=len(offs):
        plt.plot(x)
        plt.axhline(thresh)
        raise ValueError('Onsets does not match offsets')


    return(ons,offs)


def psd_integrator(y,fs,window=0.1,cutoff=2000.,nperseg=64):
    t_raw = np.arange(0,len(y))*1/fs
    f, t, Sxx = scipy.signal.spectrogram(y, fs, nperseg=nperseg)
    to_collapse = np.where(f<cutoff)[0][-1]
    P = np.mean(Sxx[:to_collapse, :], axis=0)
    frac = (window*fs)/len(t_raw)
    P2 = lowess(P, t, frac=frac)
    t2 = P2[:, 0]
    P2 = P2[:, 1]
    int_power = np.interp(t_raw, t2, P2)

    return(int_power)


def remove_EKG(x,sr,dir='pos',thresh=5):
    '''
    :param x: diaphragm (or other) emg
    :param sr: sample rate
    :return: y ekg filtered out
    '''
    if dir=='neg':
        x = -x

    xs = bwfilt(x,sr,5,500)
    pks = scipy.signal.find_peaks(xs,prominence=thresh*np.std(xs),distance=0.05*sr)[0]
    amps = xs[pks]
    win = int(0.010 *sr)
    y = x.copy()
    temp = np.zeros([2*win,len(pks)])
    for ii,pk in enumerate(pks):
        try:
            temp[:,ii] = x[pk-win:pk+win]
        except:
            pass


    if dir=='neg':
        ekg = -temp
    else:
        ekg = temp

    ekg_std = np.std(ekg[:30],0) +np.std(ekg[-30:],0)
    ekg_std = np.log(ekg_std)
    bgm = BayesianGaussianMixture(2)
    cls = bgm.fit_predict(ekg_std[:,np.newaxis])
    cls[cls==0]=-1
    m0 = np.mean(ekg_std[cls==-1])
    m1 = np.mean(ekg_std[cls==1])
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
            sm_ekg = scipy.signal.savgol_filter(ekg[:,ii],ww,1)
            y[pk - win:pk + win] -= sm_ekg
        else:
            med_ekg = np.nanmedian(ekg[:,ii-5:ii+5],1)
            med_amp = np.median(amps[ii-5:ii+5])
            scl = amps[ii]/med_amp
            y[pk - win:pk + win] -=med_ekg*scl

    return(y)


