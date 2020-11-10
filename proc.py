from scipy.signal import hilbert,savgol_filter,find_peaks
import spykes
import scipy.signal
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import numpy as np
from tqdm import tqdm
from spykes.plot import NeuroVis
import sys
import sklearn
sys.path.append('../../')
sys.path.append('../')
import utils.ephys.signal as esig

def bwfilt(x,fs,low=300,high=10000):
    b,a = scipy.signal.butter(2,[low/fs/2,high/fs/2],btype='bandpass')
    y = scipy.signal.filtfilt(b,a,x)
    return(y)

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


def get_insp_onset(nasal_trace,order=2,direction=1,thresh=3):
    """
    Return the times of inspiration onset

    :param nasal_trace: nasal thermistor trace
    :param order: Use first or second derivative (1,2)
    :param direction: do we expect inspiration to be up or down (1,-1)in the trace?
    :return: samples of inspiration onset
    """

    direction = np.sign(direction)
    d_nasal = direction*savgol_filter(np.diff(nasal_trace),501,1)
    diff2 = savgol_filter(np.diff(d_nasal),501,1)
    if order==1:
        insp_onset = find_peaks(d_nasal, prominence=np.std(d_nasal) * thresh)[0]
    elif order==2:
        insp_onset = find_peaks(diff2, prominence=np.std(diff2) * thresh)[0]
    else:
        raise NotImplemented('Only order 1 or 2 is implemented')
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


def angular_response_hist(angular_var, sp, nbins=100,min_obs=5):
    '''
    Given an angular variable that varies on -pi:pi,
    returns the probability of observing a spike (or gives a spike rate) normalized by
    the number of observations of that angular variable.

    INPUTS: angular var -- either a numpy array or a neo analog signal. Should be 1-D
            sp -- type: neo.core.SpikeTrain, numpy array. Sp can either be single spikes or a rate
    OUTPUTS:    rate -- the stimulus evoked rate at each observed theta bin
                theta_k -- the observed theta bins
                theta -- the preferred direction as determined by vector mean
                L_dir -- The preferred direction tuning strength (1-CircVar)
    '''


    if type(nbins)==int:
        bins = np.linspace(-np.pi,np.pi,nbins+1,endpoint=True)
    else:
        bins = nbins
    # not nan is a list of finite sample indices, rather than a boolean mask. This is used in computing the posterior
    not_nan = np.where(np.isfinite(angular_var))[0]
    prior,prior_edges = np.histogram(angular_var[not_nan], bins=bins)
    prior[prior < min_obs] = 0
    # allows the function to take a spike train or a continuous rate to get the posterior
    posterior, theta_k = np.histogram(angular_var[not_nan], weights=sp[not_nan], bins=bins)

    #
    rate = np.divide(posterior,prior,dtype='float32')
    theta,L_dir = get_PD_from_hist(theta_k[:-1],rate)

    return rate,theta_k,theta,L_dir


def bin_trains(ts,idx,max_time=None,binsize=0.05,start_time=5):
    '''
    bin_trains(ts,idx,n_neurons,binsize=0.05,start_time=5):
    :param ts: Array of all spike times across all neurons
    :param idx: cell index
    :param binsize:
    :param start_time:
    :return:
    '''
    if max_time is None:
        max_time = np.max(ts)

    # Keep neuron index correct
    n_neurons = np.max(idx)+1
    cell_id = np.arange(n_neurons)
    bins = np.arange(start_time, max_time, binsize)
    raster = np.empty([n_neurons, len(bins)])
    # Remove spikes that happened before the start time
    idx = idx[ts>start_time]
    ts = ts[ts>start_time]
    # Remove spikes that happened after the max time
    idx = idx[ts<max_time]
    ts = ts[ts<max_time]
    # Loop through cells
    for cell in cell_id:
        cell_ts = ts[idx==cell]
        raster[cell, :-1]= np.histogram(cell_ts, bins)[0]
    return(raster,cell_id,bins)


def get_opto_tagged(ts,pulse_on,thresh=0.25,lockout=2,max=9):
    '''

    :param ts: Spike times
    :param pulse_on: opto onset times
    :param thresh: How many spikes need to be post stimulus to count as tagges
    :param lockout: time window around onset to exclude (for light artifacts)
    :param max: max time window to consider
    :return: is_tagged: boolean if this neuron has been classified as optotagged
    '''
    neuron = spykes.NeuroVis(ts)
    df = pd.DataFrame()
    df['opto'] = pulse_on
    pre = neuron.get_spikecounts(event='opto', df=df, window=[-max, -lockout])
    post = neuron.get_spikecounts(event='opto', df=df, window=[lockout, max])

    tot_spikes = np.sum(post)

    post = np.mean(post)
    pre = np.mean(pre)
    normed_spikes = ((post-pre) / (pre + post))
    if normed_spikes>thresh:
        is_tagged = True
    else:
        is_tagged= False
    # If the number of spikes is less than 85% of the number of stimulations, do not tag
    if tot_spikes<.85*len(pulse_on):
        is_tagged=False
    return(is_tagged)


def get_event_triggered_st(ts,events,idx,pre_win,post_win):
    D = ts-events[:,np.newaxis]
    mask = np.logical_or(D<-pre_win,D>post_win)
    D[mask] = np.nan
    pop = []
    for ii in tqdm(np.unique(idx)):
        trains = []
        for jj in range(len(events)):
            sts = D[jj,idx==ii]
            sts = sts[np.isfinite(sts)]
            trains.append(sts)

        pop.append(trains)
    return(pop)


def calc_is_mod(ts,events,pre_win=-0.1,post_win=.150):
    '''
    Calculate whether the spike rate of a neuron is altered after an event

    :param ts: all spike times of a given neuron
    :param events: event times
    :param pre_win: window before event to consider (must be negative) in seconds
    :param post_win: window_after event to consider (must be positive) in seconds
    :return:
            is_mod: boolean result of wilcoxon rank sum test
            mod_depth: mean firing rate modulation (post-pre)/(pre+post)
    '''
    neuron = NeuroVis(ts)
    df= pd.DataFrame()
    df['event'] = events
    pre = neuron.get_spikecounts('event', df=df, window=[pre_win*1000, 0])
    post = neuron.get_spikecounts('event', df=df, window=[0, post_win*1000])

    # Catch instances of fewer than 100 spikes
    if np.sum(pre+post)<100:
        is_mod=False
        effect = np.nan
        return(is_mod,effect)

    # Normalize for time
    pre =pre / np.abs(pre_win)
    post=post / post_win

    # Calculate signifigance
    p = scipy.stats.wilcoxon(pre,post).pvalue

    if p<0.05:
        is_mod=True
    else:
        is_mod=False


    effect = np.nanmean((post-pre)/(post+pre))
    return(is_mod,effect)


def pop_is_mod(spiketimes,cell_id,events,**kwargs):
    '''
    wraps to calc is mod to allow population calculations of modulation by event

    :param spiketimes: array of all spike times across all cells (in seconds)
    :param cell_id: array of cell ids from which the corresponding spike in spiketimes is referenced
    :param events: times of events to consider (in seconds)
    :param kwargs: keywords passed to calc_is_mod (pre_win,post_win)
    :return:
            is_mod: boolean array of whether a given cell is modulated
            mod_depth: the modulation depth of each cell
    '''

    is_mod = np.zeros(np.max(cell_id)+1,dtype='bool')
    mod_depth = np.zeros(np.max(cell_id)+1)
    for ii,cell in enumerate(np.unique(cell_id)):
        sts = spiketimes[cell_id==cell]
        if len(sts)<10:
            continue
        cell_is_mod,cell_mod_depth = calc_is_mod(sts,events,**kwargs)
        is_mod[ii] = cell_is_mod
        mod_depth[ii] = cell_mod_depth

    mod_depth[np.isnan(mod_depth)] = 0

    return(is_mod,mod_depth)


def proc_pleth(pleth,tvec,width=0.01,prominence=0.3,height = 0.3,distance=0.1):
    '''
    Calculates the inspiration onsets and offsets.
    Only looks at positive deflections in the pleth
    Takes find_peaks kwargs

    :param pleth: Plethysmography data
    :param tvec: Vector mapping samples to timestamps
    :param width: Minimum width that the signal has to be above threshold to be considered an inspiration (s)
    :param prominence: Miniumum prominence needed (see find_peaks) (v)
    :param height: Minimum absolute height (v)
    :param distance: Minimum time between inspirations (s)
    :return:
            pleth_on_t - timestamps of pleth onsets
            pleth_data - dictionary of inspiration parameters:
                on_samp:        Sample index of pleth onsets
                off_samp:       Sample index of pleth onsets
                amp:            Amplitude of pleth peak
                duration_sec:   Duration of a pleth inspiration in seconds
                duration_samp:  Duration of a pleth inspiration in seconds
    '''

    ## keep only positive pleth values to get inspirations
    temp_pleth = pleth.copy()
    temp_pleth[temp_pleth<0] = 0

    # Sampling rate is the difference of the first 2 time samples
    sr = 1/(tvec[1]-tvec[0])

    # Get pleth peaks
    pk_pleth = scipy.signal.find_peaks(temp_pleth,width=width*sr,prominence=prominence,height=height,distance=distance*sr)[0]
    pleth_on,pleth_off = scipy.signal.peak_widths(temp_pleth,pk_pleth,rel_height=0.9)[2:]

    # Map the on and off to ints to allow for indexing
    pleth_on = pleth_on.astype('int')
    pleth_off = pleth_off.astype('int')

    # Map indices to values
    pleth_on_t = tvec[pleth_on]
    pleth_off_t = tvec[pleth_off]
    pleth_amp = pleth[pk_pleth]

    pleth_data = {}
    pleth_data['on_samp'] = pleth_on
    pleth_data['off_samp'] = pleth_off
    pleth_data['on_sec'] = pleth_on_t
    pleth_data['off_sec'] = pleth_off_t
    pleth_data['amp'] = pleth_amp
    pleth_data['duration_sec'] = pleth_off_t-pleth_on_t
    pleth_data['duration_samp'] = pleth_off-pleth_on
    pleth_data['pk_samp'] = pk_pleth.astype('int')
    pleth_data['pk_time'] = tvec[pk_pleth.astype('int')]
    pleth_data['postBI'] = np.hstack([pleth_on_t[1:]-pleth_off_t[:-1],[np.nan]])

    return(pleth_on_t,pleth_data)


def events_to_rate(evt,max_time,dt,start_time=0):
    '''

    :param evt: array of times (in seconds) in which events occur
    :param max_time: last time to map to
    :param dt: time step to use between updates of rate
    :return:
            tvec - a vector of timestamps with the chosen dt
            rate - the value of the rate over those timestamps
    '''
    tvec = np.arange(start_time,max_time,dt)
    rate = np.zeros_like(tvec)
    last_val = 0
    for ii in range(1,len(evt)):
        t0 = evt[ii-1]
        tf = evt[ii]
        rr = tf-t0
        next_val = np.searchsorted(tvec,tf)
        rate[last_val:next_val] = 1/rr
        last_val = next_val

    rate[last_val:]=1/rr
    return(tvec,rate)


def proc_dia(dia,sr,qrs_thresh=6,dia_thresh=1):
    '''
    Processes the raw diaphragm by performing filtering, EKG removal, rectification
    integration (medfilt), burst detection, and burst quantification
    :param dia: raw diaphragm recording
    :param sr: sample rate
    :param qrs_thresh: threshold (standardized) to detect the ekg signal (default=6)
    :param dia_thresh: threshold (standardized) to detect diaphragm recruitment
    :return:
            dia_df - DataFrame with various burst features as derived from the diaphragm
            phys_df - DataFrame with physiology data - heart rate and diaphragm rate
    '''

    # Window for time of QRS shapes
    win = int(0.010 *sr)
    max_t = len(dia)/sr
    # Bandpass filter the recorded diaphragm
    xs = esig.bwfilt(dia,sr,10,10000)
    # Get QRS peak times
    pulse = scipy.signal.find_peaks(xs,prominence=qrs_thresh*np.std(xs),distance=0.05*sr)[0]
    # Create a copy of the smoothed diaphragm - may not be needed
    y = xs.copy()
    # Preallocate for all QRS shapes
    QRS = np.zeros([2*win,len(pulse)])

    # Get each QRS complex
    for ii,pk in enumerate(pulse):
        try:
            QRS[:,ii] = xs[pk-win:pk+win]
        except:
            pass
    # Replace each QRS complex with the average of ten nearby QRS
    for ii,pk in enumerate(pulse):
        if pk-win<0:
            continue
        if (pk+win)>len(xs):
            continue
        xs[pk-win:pk+win] -= np.nanmean(QRS[:,ii-5:ii+5],1)

    pulse_times = pulse/sr

    xs[np.isnan(xs)] = 0
    xss = esig.bwfilt(xs,sr,1000,10000)
    smooth_win = int(0.05*sr)

    print('Integrating, this can take a while')
    integrated = np.sqrt(scipy.signal.medfilt(xss**2,smooth_win+1))
    print('Integrated!')

    scl = sklearn.preprocessing.StandardScaler(with_mean=0)
    integrated_scl = scl.fit_transform(integrated[:,np.newaxis]).ravel()

    pks = scipy.signal.find_peaks(integrated_scl,
                                  prominence=dia_thresh,
                                  distance=int(0.200*sr),
                                  width=int(0.050*sr))[0]
    lips = scipy.signal.peak_widths(integrated,pks,rel_height=0.9)[2]
    rips = scipy.signal.peak_widths(integrated,pks,rel_height=0.8)[3]
    lips = lips.astype('int')
    rips = rips.astype('int')

    amp = np.zeros(len(lips))
    auc = np.zeros(len(lips))
    for ii,(lip,rip) in enumerate(zip(lips,rips)):
        temp = integrated[lip:rip]
        amp[ii] = np.percentile(temp,95)
        auc[ii] = np.trapz(temp)
    dur = rips-lips

    lips_t = lips/sr
    rips_t = rips/sr

    dia_data = {}
    dia_data['on_samp'] = lips
    dia_data['off_samp'] = rips
    dia_data['on_sec'] = lips_t
    dia_data['off_sec'] = rips_t
    dia_data['amp'] = amp
    dia_data['auc'] = auc
    dia_data['duration_sec'] = dur/sr
    dia_data['duration_samp'] = dur
    dia_data['pk_samp'] = pks
    dia_data['pk_time'] = pks/sr
    dia_data['postBI'] = np.hstack([lips_t[1:]-rips_t[:-1],[np.nan]])

    dia_df = pd.DataFrame(dia_data)

    rate_t,dia_rate = events_to_rate(lips_t,max_t,0.1)
    rate_t,pulse_rate = events_to_rate(pulse_times,max_t,0.1)
    pulse_rate = scipy.signal.medfilt(pulse_rate,5)

    phys_df = pd.DataFrame()
    phys_df['t'] = rate_t
    phys_df['heart_rate'] = pulse_rate
    phys_df['dia_rate'] = dia_rate



    return(dia_df,phys_df)






