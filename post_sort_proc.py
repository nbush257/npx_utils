import warnings
from scipy.signal import hilbert,savgol_filter,find_peaks
import numpy as np
import pandas as pd
import math

from scipy.signal import butter,filtfilt,lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    Helper function to create butterworth filter parameters
    Parameters
    ----------
    lowcut :    lowcut frequency in Hz
    highcut :   highcut frequency in Hz
    fs :        sampling rate in Hz
    order :     butterwoth filter order (int)

    Returns
    -------
    b,a - Parameters for a butterworth filter

    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Wrapper to
    Parameters
    ----------
    data :      an aarray of data to be filtered
    lowcut :    lowcut frequency in Hz
    highcut :   highcut frequency in Hz
    fs :        sampling rate in Hz
    order :     butterwoth filter order (int)

    Returns
    -------
    y : filtered data

    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


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


def nasal_to_phase(x):
    '''
    Given an input array, use a Hilbert transform to return the phase of that signal
    over time.

    This implementation is designed for very large array as the hilbert chokes on too large a vector.
    Chunks vector into overlapping windows and manipulates overlaps to create smooth transitions between windows

    xa_tot is computed but not returned.

    Parameters
    ----------
    x : input data to be transformed

    Returns
    -------
    phi : vector of phase values on the interval [-pi:pi] over time
    '''

    x = savgol_filter(x, 101, 2)
    window = 500000
    overlap = 20000
    toss = 10000
    n_wins = np.ceil(x.shape[0] / window).astype('int')
    phi = np.empty(x.shape[0]) * np.nan
    xa_tot = np.empty(x.shape[0],dtype='complex')

    for win in range(n_wins):
        if win == 0:
            start = win * window
            stop = start + window + overlap
        else:
            start = win * window - overlap
            stop = start + window + overlap * 2

        xx = x[start:stop]
        xa = hilbert(xx)

        if win == 0:
            phi[start:stop - toss] = np.angle(xa[:-toss])
            xa_tot[start:stop - toss] = xa[:-toss]
        elif win == n_wins:
            phi[start + overlap - toss:stop] = np.angle(xa[overlap - toss:])
            xa_tot[start + overlap - toss:stop] = xa[overlap - toss:]
        else:
            phi[start + overlap:stop] = np.angle(xa[overlap:])
            xa_tot[start + overlap:stop] = xa[overlap:]
    return(phi)


def get_trains_seconds(spike_extractor):
    #TODO: remove
    trains = spike_extractor.get_units_spike_train()
    trains_seconds = []
    for train in trains:
        trains_seconds.append(train / spike_extractor.get_sampling_frequency())
    return(trains_seconds)


def align_spikes_to_nasal(nasal,spike_extractor):
    '''
    creates in place boolean mask for spikes
    :param nasal:
    :param spike_extractor:
    :return:
    '''
    #TODO: remove the spikeectractor element?
    trains_seconds = get_trains_seconds(spike_extractor)
    for ii,train in enumerate(trains_seconds):
        nasal[f'U{ii:03d}'] = np.zeros(nasal.shape[0],dtype='int')
        for spike in train:
            nasal_idx  = find_nearest(nasal.index,spike)
            nasal[f'U{ii:03d}'][nasal_idx] = 1


def get_insp_onset(nasal_trace):
    '''
    Return the times of inspiration onset

    :param nasal_trace:
    :return insp_onset: returns the index in nasal_trace of the inspiration onsets
    '''
    smooth = savgol_filter(nasal_trace, 101, 1)
    diff2 = np.diff(savgol_filter(np.diff(smooth), 101, 1))
    insp_onset = find_peaks(diff2, prominence=np.std(diff2) * 3)[0]
    return(insp_onset)


def shift_phi(phi,insp_onset):
    '''
    Shifts the phi such that 0 is inspiration onset
    :param nasal:
    :return new_phi: phase shifted phi
    '''
    m_phi = np.mean(phi[insp_onset])
    new_phi = phi.copy()
    new_phi-=m_phi
    new_phi = np.unwrap(new_phi)
    new_phi = (new_phi + np.pi) % (2 * np.pi) - np.pi

    return(new_phi)


def get_LFP(rec,chan_ids):
    '''
    Not a good function! placeholder until we get better
    I/O modules
    :param rec:
    :param chan_ids:
    :return:
    '''
    #TODO: remove
    warnings.warn('You really shouldnt be using this')
    raw_dat = pd.read_hdf(rec,'SPKC')
    ad_gain = pd.read_hdf(rec,'ad_gain')

    all_LFP = pd.DataFrame()
    for chan in chan_ids:
        raw = raw_dat.iloc[:, chan]
        raw *= ad_gain.values[0][0]
        LFP = butter_bandpass_filter(raw.values.ravel(),0.1,60.,1/raw.index[1],order=2)
        all_LFP[chan] = LFP

    all_LFP.index = raw_dat.index
    all_LFP = all_LFP.iloc[::8,:]
    return(all_LFP)



def get_PD_from_hist(theta_k,rate):
    '''
    Calculate the vector sum direction and tuning strength from a histogram of responses in polar space

    Migrated from whisker work, but should generalize to any polar histogram
    INPUTS: theta_k -- sampled bin locations from a polar histogram.
                        * Assumes number of bins is the same as the number of observed rates (i.e. if you use bin edges you will probably have to truncate the input to fit)
                        * Bin centers is a better usage
            rate -- observed rate at each bin location
    OUTPUTS:    theta -- the vector mean direction of the input bin locations and centers
                L_dir -- the strength of the tuning as defined by Mazurek FiNC 2014. Equivalient to 1- Circular Variance
    '''

    # Calculate the direction tuning strength
    L_dir = np.abs(
        np.sum(
            rate * np.exp(1j * theta_k)) / np.sum(rate)
    )

    # calculate vector mean
    x = rate * np.cos(theta_k)
    y = rate * np.sin(theta_k)

    X = np.sum(x) / len(x)
    Y = np.sum(y) / len(x)

    theta = np.arctan2(Y, X)

    return theta,L_dir


def angular_response_hist(angular_var, sp, use_flags, nbins=100,min_obs=5):
    '''
    Given an angular variable that varies on -pi:pi,
    returns the probability of observing a spike (or gives a spike rate) normalized by
    the number of observations of that angular variable.

    Migrated from whisker work. May need to be refactored to be more general

    INPUTS: angular var -- either a numpy array or a neo analog signal. Should be 1-D
            sp -- type: neo.core.SpikeTrain, numpy array. Sp can either be single spikes or a rate
    OUTPUTS:    rate -- the stimulus evoked rate at each observed theta bin
                theta_k -- the observed theta bins
                theta -- the preferred direction as determined by vector mean
                L_dir -- The preferred direction tuning strength (1-CircVar)
    '''

    # Overloaded for neo object or array
    if type(angular_var)==neo.core.analogsignal.AnalogSignal:
        angular_var = angular_var.magnitude
    if angular_var.ndim==2:
        if angular_var.shape[1] == 1:
            angular_var = angular_var.ravel()
        else:
            raise Exception('Angular var must be able to be unambiguously converted into a vector')

    if type(nbins)==int:
        bins = np.linspace(-np.pi,np.pi,nbins+1,endpoint=True)
    # not nan is a list of finite sample indices, rather than a boolean mask. This is used in computing the posterior
    not_nan = np.where(np.logical_and(np.isfinite(angular_var),use_flags))[0]
    prior,prior_edges = np.histogram(angular_var[not_nan], bins=bins)
    prior[prior < min_obs] = 0
    # allows the function to take a spike train or a continuous rate to get the posterior
    if type(sp)==neo.core.spiketrain.SpikeTrain:
        spt = sp.times.magnitude.astype('int')
        idx = [x for x in spt if x in not_nan]
        posterior, theta_k = np.histogram(angular_var[idx], bins=bins)
    else:
        posterior, theta_k = np.histogram(angular_var[not_nan], weights=sp[not_nan], bins=bins)

    #
    rate = np.divide(posterior,prior,dtype='float32')
    theta,L_dir = get_PD_from_hist(theta_k[:-1],rate)

    return rate,theta_k,theta,L_dir





