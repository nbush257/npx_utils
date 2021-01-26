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
    b,a = scipy.signal.butter(4,[low/fs/2,high/fs/2],btype='bandpass')
    y = scipy.signal.filtfilt(b,a,x)
    return(y)


def calc_phase(x):
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


def shift_phi(phi,insp_onset):
    '''
    Shifts the phi such that 0 is inspiration onset
    :param phi: phase trace
    :param insp_onset: samples to shift to phi=0
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


def compute_KL(var,sp,nbins=25,min_obs=5):
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


    # not nan is a list of finite sample indices, rather than a boolean mask. This is used in computing the posterior
    not_nan = np.where(np.isfinite(var))[0]
    prior, prior_edges = np.histogram(var[not_nan], bins=nbins)
    prior[prior < min_obs] = 0
    # allows the function to take a spike train or a continuous rate to get the posterior
    posterior, theta_k = np.histogram(var[not_nan], weights=sp[not_nan], bins=nbins)

    KL = scipy.stats.entropy(posterior,prior)
    return(KL)


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
    # If the number of spikes is less than 75% of the number of stimulations, do not tag
    if tot_spikes<.75*len(pulse_on):
        is_tagged=False
    return(is_tagged)


def get_event_triggered_st(ts,idx,events,pre_win,post_win):
    '''
    Calculate the spike times that occurred before and after a given lis of events
    and return a list of neo spike trains
    :param ts: array of spike times
    :param idx: cell_id associated with the unit number (must be of same length as ts)
    :param events: array of event times
    :param pre_win: window of time prior to event (in seconds)
    :param post_win: window of event after event
    :return pop: a list of neo spike trains
    '''
    assert(len(ts)==len(idx))
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


def events_to_rate(evt,max_time,dt,start_time=0):
    '''
    Calculate a time vector and a rate vector from discrete events.
    Useful for mapping things like respiratory rate or heart rate

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


def proc_pleth(pleth,sr,width=0.01,prominence=0.3,height = 0.3,distance=0.1):
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


    # Get pleth peaks
    pk_pleth = scipy.signal.find_peaks(temp_pleth,width=width*sr,prominence=prominence,height=height,distance=distance*sr)[0]
    pleth_on,pleth_off = scipy.signal.peak_widths(temp_pleth,pk_pleth,rel_height=0.9)[2:]

    # Map the on and off to ints to allow for indexing
    pleth_on = pleth_on.astype('int')
    pleth_off = pleth_off.astype('int')

    # Map indices to values
    pleth_on_t = pleth_on/sr
    pleth_off_t = pleth_off/sr
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
    pleth_data['pk_time'] = pk_pleth.astype('int')/sr
    pleth_data['postBI'] = np.hstack([pleth_on_t[1:]-pleth_off_t[:-1],[np.nan]])
    pleth_df = pd.DataFrame(pleth_data)

    return(pleth_df)


def proc_dia(dia,sr,qrs_thresh=6,dia_thresh=1,method='triang',win=0.05):
    '''
    Processes the raw diaphragm by performing filtering, EKG removal, rectification
    integration, burst detection, and burst quantification
    :param dia: raw diaphragm recording
    :param sr: sample rate
    :param qrs_thresh: threshold (standardized) to detect the ekg signal (default=6)
    :param dia_thresh: threshold (standardized) to detect diaphragm recruitment
    :param method: Either a scipy.signal.window string, or 'med' (warning- med is sloowww)
    :return:
            dia_df - DataFrame with various burst features as derived from the diaphragm
            phys_df - DataFrame with physiology data - heart rate and diaphragm rate
            integrated - array of integrated diaphragm
    '''
    max_t = len(dia)/sr
    integrated,pulse_times = integrate_dia(dia,sr,qrs_thresh=qrs_thresh,method=method,win=win)
    dia_df = burst_stats_dia(integrated,sr,dia_thresh=dia_thresh)

    rate_t,dia_rate = events_to_rate(dia_df['on_sec'],max_t,0.1)
    rate_t,pulse_rate = events_to_rate(pulse_times,max_t,0.1)
    pulse_rate = scipy.signal.medfilt(pulse_rate,11)

    phys_df = pd.DataFrame()
    phys_df['t'] = rate_t
    phys_df['heart_rate'] = pulse_rate
    phys_df['dia_rate'] = dia_rate
    phys_df = phys_df.set_index('t')
    return(dia_df,phys_df,integrated)


def integrate_dia(dia,sr,qrs_thresh=6,method='triang',win=0.05):
    '''
    Remove EKG and integrate the diaphragm trace
    :param dia: raw diaphragm recording
    :param sr: sample rate
    :param qrs_thresh: threshold (standardized) to detect the ekg signal (default=6)
    :param method: method by which to integrate ['med'...] 'med' is very slow, but good. Takes any valid argument to scipy.signal.get_window
    :return:
            integrated - integrated diaphragm trace
            pulse_times - heartbeat times
    '''
    # Window for time of QRS shapes
    win_qrs = int(0.010 *sr)
    # Bandpass filter the recorded diaphragm
    xs = esig.bwfilt(dia,sr,10,10000)
    # Get QRS peak times
    pulse = scipy.signal.find_peaks(xs,prominence=qrs_thresh*np.std(xs),distance=0.05*sr)[0]
    # Create a copy of the smoothed diaphragm - may not be needed
    y = xs.copy()
    # Preallocate for all QRS shapes
    QRS = np.zeros([2*win_qrs,len(pulse)])

    # Get each QRS complex
    for ii,pk in enumerate(pulse):
        try:
            QRS[:,ii] = xs[pk-win_qrs:pk+win_qrs]
        except:
            pass
    # Replace each QRS complex with the average of ten nearby QRS
    for ii,pk in enumerate(pulse):
        if pk-win_qrs<0:
            continue
        if (pk+win_qrs)>len(xs):
            continue
        xs[pk-win_qrs:pk+win_qrs] -= np.nanmean(QRS[:,ii-5:ii+5],1)

    pulse_times = pulse/sr

    xs[np.isnan(xs)] = 0
    xss = esig.bwfilt(xs,sr,1000,10000)

    if method == 'med':
        smooth_win = int(win* sr)
        print('Integrating, this can take a while')
        integrated = np.sqrt(scipy.signal.medfilt(xss**2,smooth_win+1))
        print('Integrated!')
    else:
        smooth_win = scipy.signal.get_window(method,int(win * sr))
        integrated = np.sqrt(scipy.signal.convolve(xss**2,smooth_win,'same'))/len(smooth_win)
    return(integrated,pulse_times)


def burst_stats_dia(integrated,sr,dia_thresh=1):
    '''
    Calculate diaphragm burst features
    :param integrated: integrated diaphragm trace
    :param sr: sample rate
    :param dia_thresh:
    :return:
            dia_df - dataframe of diaphragm burst features
    '''
    scl = sklearn.preprocessing.StandardScaler(with_mean=0)
    integrated_scl = scl.fit_transform(integrated[:,np.newaxis]).ravel()

    pks = scipy.signal.find_peaks(integrated_scl,
                                  prominence=dia_thresh,
                                  distance=int(0.200*sr),
                                  width=int(0.050*sr))[0]
    lips = scipy.signal.peak_widths(integrated,pks,rel_height=0.8)[2]
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

    return(dia_df)


def events_in_epochs(evt,epoch_times,epoch_labels=None):
    '''
    Given a list of events, categorizes which epoch they are in from the list of epochs
    Useful for labelling dataframes of onset events with things like "normoxia"
    :param evt:
    :param epoch_times: include 0
    :param epoch_labels:
    :return:
    '''
    cat = np.zeros(len(evt))
    # loop through
    for ii in range(len(epoch_times)-1):
        lb = epoch_times[ii]
        ub = epoch_times[ii+1]
        mask = np.logical_and(
            evt>lb,evt<ub
        )
        cat[mask]=ii

    lb = epoch_times[-1]
    ub = np.Inf

    mask = np.logical_and(
        evt > lb, evt < ub
    )
    cat[mask] = ii+1

    return(cat)


def jitter(data,l):
    """
    Jittering multidemntational logical data where
    0 means no spikes in that time bin and 1 indicates a spike in that time bin.
     Be sure to cite Xiaoxuan Jia https://github.com/jiaxx/jitter
    """
    if len(np.shape(data)) > 3:
        flag = 1
        sd = np.shape(data)
        data = np.reshape(data, (
        np.shape(data)[0], np.shape(data)[1], len(data.flatten()) / (np.shape(data)[0] * np.shape(data)[1])),
                          order='F')
    else:
        flag = 0

    psth = np.mean(data, axis=1)
    length = np.shape(data)[0]

    if np.mod(np.shape(data)[0], l):
        data[length:(length + np.mod(-np.shape(data)[0], l)), :, :] = 0
        psth[length:(length + np.mod(-np.shape(data)[0], l)), :] = 0

    if np.shape(psth)[1] > 1:
        dataj = np.squeeze(
            np.sum(np.reshape(data, [l, np.shape(data)[0] // l, np.shape(data)[1], np.shape(data)[2]], order='F'),
                   axis=0))
        psthj = np.squeeze(
            np.sum(np.reshape(psth, [l, np.shape(psth)[0] // l, np.shape(psth)[1]], order='F'), axis=0))
    else:
        dataj = np.squeeze(np.sum(np.reshape(data, l, np.shape(data)[0] // l, np.shape(data)[1], order='F')))
        psthj = np.sum(np.reshape(psth, l, np.shape(psth)[0] // l, order='F'))

    if np.shape(data)[0] == l:
        dataj = np.reshape(dataj, [1, np.shape(dataj)[0], np.shape(dataj)[1]], order='F');
        psthj = np.reshape(psthj, [1, np.shape(psthj[0])], order='F');

    psthj = np.reshape(psthj, [np.shape(psthj)[0], 1, np.shape(psthj)[1]], order='F')
    psthj[psthj == 0] = 10e-10

    corr = dataj / np.tile(psthj, [1, np.shape(dataj)[1], 1]);
    corr = np.reshape(corr, [1, np.shape(corr)[0], np.shape(corr)[1], np.shape(corr)[2]], order='F')
    corr = np.tile(corr, [l, 1, 1, 1])
    corr = np.reshape(corr, [np.shape(corr)[0] * np.shape(corr)[1], np.shape(corr)[2], np.shape(corr)[3]],
                      order='F');

    psth = np.reshape(psth, [np.shape(psth)[0], 1, np.shape(psth)[1]], order='F');
    output = np.tile(psth, [1, np.shape(corr)[1], 1]) * corr

    output = output[:length, :, :]
    return output


def xcorrfft(a,b,NFFT):
    CCG = np.fft.fftshift(np.fft.ifft(np.multiply(np.fft.fft(a,NFFT), np.conj(np.fft.fft(b,NFFT)))))
    return CCG


def nextpow2(n):
    """get the next power of 2 that's greater than n"""
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


def get_ccgjitter(spikes, FR, jitterwindow=25):
    # spikes: neuron*ori*trial*time
    assert np.shape(spikes)[0]==len(FR)

    n_unit=np.shape(spikes)[0]
    n_t = np.shape(spikes)[3]
    # triangle function
    t = np.arange(-(n_t-1),(n_t-1))
    theta = n_t-np.abs(t)
    del t
    NFFT = int(nextpow2(2*n_t))
    target = np.array([int(i) for i in NFFT/2+np.arange((-n_t+2),n_t)])

    ccgjitter = []
    pair=0
    for i in np.arange(n_unit-1): # V1 cell
        for m in np.arange(i+1,n_unit):  # V2 cell
            if FR[i]>2 and FR[m]>2:
                temp1 = np.squeeze(spikes[i,:,:,:])
                temp2 = np.squeeze(spikes[m,:,:,:])
                FR1 = np.squeeze(np.mean(np.sum(temp1,axis=2), axis=1))
                FR2 = np.squeeze(np.mean(np.sum(temp2,axis=2), axis=1))
                tempccg = xcorrfft(temp1,temp2,NFFT)
                tempccg = np.squeeze(np.nanmean(tempccg[:,:,target],axis=1))

                temp1 = np.rollaxis(np.rollaxis(temp1,2,0), 2,1)
                temp2 = np.rollaxis(np.rollaxis(temp2,2,0), 2,1)
                ttemp1 = jitter(temp1,jitterwindow)
                ttemp2 = jitter(temp2,jitterwindow)
                tempjitter = xcorrfft(np.rollaxis(np.rollaxis(ttemp1,2,0), 2,1),np.rollaxis(np.rollaxis(ttemp2,2,0), 2,1),NFFT);
                tempjitter = np.squeeze(np.nanmean(tempjitter[:,:,target],axis=1))
                ccgjitter.append((tempccg - tempjitter).T/np.multiply(np.tile(np.sqrt(FR[i]*FR[m]), (len(target), 1)),
                    np.tile(theta.T.reshape(len(theta),1),(1,len(FR1)))))

    ccgjitter = np.array(ccgjitter)
    return ccgjitter
