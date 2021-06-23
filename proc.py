import neo
import quantities
from scipy.signal import hilbert,savgol_filter
import spykes
import scipy.signal
import scipy.stats
import pandas as pd
import numpy as np
from tqdm import tqdm
from spykes.plot import NeuroVis
import sys
import sklearn
import scipy.ndimage
import elephant
import quantities as pq
import warnings
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')


def bwfilt(x,fs,low=300,high=10000):
    '''
    Convenience function to a 4th order butterworth bandpass filter
    :param x: input vector
    :param fs: sample rate (Hz)
    :param low: (Hz)
    :param high: (Hz)
    :return: y - filtered vector
    '''
    b,a = scipy.signal.butter(4,[low/fs/2,high/fs/2],btype='bandpass')
    y = scipy.signal.filtfilt(b,a,x)
    return(y)


def calc_phase(x,sr):
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
    sgolay_win = sr*.01 # 10ms
    if sgolay_win %2 ==0:
        sgolay_win+=1
    sgolay_win = int(sgolay_win)
    x = savgol_filter(x, sgolay_win, 2)
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


def calc_dia_phase(ons,offs=None,t_start=0,t_stop=None,dt=1/1000):
    '''
    Computes breathing phase based on the diaphragm
    Phase is [0,1] where 0 is diaphragm onset, 0.5 is diaphragm offset, and 1 is diaphragm onset again,
     - NB: Technically can generalize to any on/off signal, but standard usage should be diaphragm
    :param ons: timestamps of diaphragm onsets (sec)
    :param offs: timestamps of diaphragm offsets (sec). If no offs is given, linearly spaces onsets
    :param t_start: start time of the phase trace (default=0)
    :param t_stop: stop time of the phase trace (default is last stop value)
    :param dt: time between timesteps(set to 1kHz)
    :return:
            phi - phase over time
            t_phi - timestamps of the phase vector
    '''
    if t_stop is None:
        t_stop = offs[-1]
    if t_stop<t_start:
        raise ValueError(f'Stop time: {t_stop}s cannot be less than start time: {t_start}s')

    offs = offs[offs>t_start]
    offs = offs[offs<=t_stop]

    ons = ons[ons>=t_start]
    ons = ons[:len(offs)]


    assert(len(ons)==len(offs))
    assert(np.all(np.greater(offs,ons)))

    t_phi =np.arange(t_start,t_stop,dt)
    phi = np.zeros_like(t_phi)

    n_breaths = len(ons)

    if offs is not None:
        for ii in range(n_breaths-1):
            on = ons[ii]
            off = offs[ii]
            next_on = ons[ii+1]
            idx = np.searchsorted(t_phi,[on,off,next_on])
            phi[idx[0]:idx[1]] = np.linspace(0,0.5,idx[1]-idx[0])
            try:
                phi[idx[1]:idx[2]] = np.linspace(0.5,1,idx[2]-idx[1])
            except:
                print([on,off,next_on])
                print(idx)
                print(ii)

    else:
        for ii in range(n_breaths-1):
            on = ons[ii]
            next_on = ons[ii+1]
            idx = np.searchsorted(t_phi,[on,next_on])
            phi[idx[0]:idx[1]] = np.linspace(0,1,idx[1]-idx[0])

    return(t_phi,phi)


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


def hist_tuning(x,x_t,st,bins=50):
    '''
    Get the normalized tuning curve for a non-angular
    value
    :param x: Covariate to tune to
    :param x_t: the time stamps of x
    :param st: spike times
    :param bins: bins for the histogram
    :return:
    '''
    prior,bins = np.histogram(x,bins=bins)
    idx = np.searchsorted(x_t,st)
    conditional,bins = np.histogram(x[idx],bins=bins)
    posterior = conditional/prior

    return(posterior,bins[1:])


def bin_trains(ts,idx,max_time=None,binsize=0.05,start_time=5):
    '''
    bin_trains(ts,idx,n_neurons,binsize=0.05,start_time=5):
    :param ts: Array of all spike times across all neurons
    :param idx: cell index
    :param binsize:
    :param start_time:
    :return: raster,cell_id,bins
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


def burst_stats_dia(integrated,sr,dia_thresh=1,rel_height=0.8):
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
                                  distance=int(0.100*sr),
                                  width=int(0.025*sr))[0]
    lips = scipy.signal.peak_widths(integrated,pks,rel_height=rel_height)[2]
    rips = scipy.signal.peak_widths(integrated,pks,rel_height=rel_height)[3]
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
    dia_df = dia_df.eval('inst_freq=1/(duration_sec+postBI)')
    dia_df = dia_df.eval('IBI=duration_sec+postBI')



    return(dia_df)


def events_in_epochs(evt,epoch_times,epoch_labels=None):
    '''
    Given a list of events, categorizes which epoch they are in from the list of epochs
    Useful for labelling dataframes of onset events with things like "normoxia"
    :param evt:
    :param epoch_times: include 0
    :param epoch_labels: list of strings to label each epoch
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

    labels = None
    if epoch_labels is not None:
        labels = [epoch_labels[int(x)] for x in cat]

    return(cat,labels)


def compute_breath_type(breaths,thresh=7):
    '''
    Labels each breath as a 'eupnea','sigh', or 'apnea' based on the
    rolling average.
    Requires the breaths dataframe in the standard preprocessing
    :param breaths:
    :return:
    '''
    temp = breaths.copy()

    filt_breaths = temp-temp.rolling(50).median()
    MAD = lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))
    rolling_MAD = temp.rolling(window=51, center=True).apply(MAD)*thresh
    idx = filt_breaths['auc']>rolling_MAD['auc']
    temp['type'] = 'eupnea'
    temp.loc[idx,'type'] = 'sigh'

    # prevent erroring out if there is no pleth analysis
    if 'inhale_onsets' in temp.columns:
        apnea_mask = temp['inhale_onsets'].isna()
        temp.loc[apnea_mask,'type']= 'apnea'

    return temp


def get_breaths(breaths,sr,analog,t_pre=0.2,t_post=0.5):
    '''
    Extracts the slice of analog data before and after each eupnea, apnea, and sigh
    :param breaths: breaths dataframe
    :param sr: sample rate (Hz)
    :param analog: data to slice
    :param t_pre: default= 0.2s
    :param t_post: default = 0.5s
    :return: eupnea, sigh, apnea, t
    '''
    t_pre_samp = int(t_pre*sr)
    t_post_samp = int(t_post*sr)
    counts = breaths.groupby('type').count()['on_sec']
    e_count=0
    a_count=0
    s_count=0

    eup = np.zeros([t_pre_samp + t_post_samp,counts['eupnea']])
    apnea = np.zeros([t_pre_samp + t_post_samp,counts['apnea']])
    sigh = np.zeros([t_pre_samp + t_post_samp,counts['sigh']])
    t = np.arange(-t_pre,t_post,1/sr)[:eup.shape[0]]

    for ii,data in breaths[['on_sec','type']].iterrows():
        on = data['on_sec']
        type = data['type']
        on_samp = int(on*sr)
        if on_samp-t_pre_samp<0:
            continue
        if on_samp+t_post_samp>len(analog):
            continue
        ana_slice = analog[on_samp-t_pre_samp:on_samp+t_post_samp]

        if type=='eupnea':
            eup[:,e_count] = ana_slice
            e_count+=1
        elif type=='apnea':
            apnea[:,a_count] = ana_slice
            a_count+=1
        elif type=='sigh':
            sigh[:,s_count] = ana_slice
            s_count+=1
        else:
            pass

    return(eup,sigh,apnea,t)


def get_sta(x,tvec,ts,win=0.5):
    '''
    Compute the spike triggered average, std, sem of a covariate x
    :param x: The analog signal to check against
    :param tvec: the time vector for x
    :param ts: the timestamps (in seconds) of the cell to get the sta for
    :param win: the window (symmetric about the spike) to average
    :return:
    '''
    assert(len(tvec)==len(x))

    dt = tvec[1]-tvec[0]
    samps = np.searchsorted(tvec,ts)
    win_samps = int(win/dt)
    spike_triggered = np.zeros([win_samps*2,len(samps)])
    for ii,samp in enumerate(samps):
        if (samp-win_samps)<0:
            continue
        if (samp+win_samps)>len(x):
            continue
        spike_triggered[:,ii] = x[samp-win_samps:samp+win_samps]

    st_average = np.nanmean(spike_triggered,1)
    st_sem = np.nanstd(spike_triggered,1)/np.sqrt(len(samps))
    st_std = np.nanstd(spike_triggered,1)
    win_t = np.linspace(-win,win,win_samps*2)
    sta = {'mean':st_average,
           'sem':st_sem,
           'std':st_std,
           't':win_t}

    return(sta)


def get_coherence(ts,x,x_sr,t0,tf):
    '''
    Compute the coherence from spike times and an analog signal
    :param ts: spiketimes (in s). Can be a quantity
    :param x: the analog signal to compare against
    :param x_sr: the sampling rate of the analog signal
    :param t0: the first time to consider (in s). Can be a quantity
    :param tf: the last time to consider (in s). Can be a quantity
    :return:    coh - the maximum coherence
                freqs - the frequencies that make up the coherence plot
                sfc - the coherences at each frequency in freqs
    '''
    if type(t0) is not pq.quantity.Quantity:
        t0 = t0*pq.s
    if type(tf) is not pq.quantity.Quantity:
        tf = tf*pq.s
    if type(ts) is not pq.quantity.Quantity:
        ts = ts*pq.s

    s0 = int(t0*x_sr)
    sf = int(tf*x_sr)
    x_slice = x[s0:sf]
    sig = neo.AnalogSignal(x_slice, units='V', sampling_rate=x_sr * pq.Hz,t_start=t0)
    ts = ts[ts<sig.t_stop]
    ts = ts[ts>sig.t_start]
    spt = neo.SpikeTrain(ts,t_start = sig.t_start,t_stop=sig.t_stop,units=pq.s)
    sfc, freqs = elephant.sta.spike_field_coherence(sig, spt, nperseg=8192)
    coh = np.max(sfc.magnitude)

    return(coh,sfc,freqs)



def get_coherence_all(spikes,x,x_sr,t0,tf,method='static'):
    '''
    wrapper to get_coherence to run on all cells in spikes
    :param ts: spiketimes (in s). Can be a quantity
    :param x: the analog signal to compare against
    :param x_sr: the sampling rate of the analog signal
    :param t0: the first time to consider (in s). Can be a quantity
    :param tf: the last time to consider (in s). Can be a quantity
    :return: coh_df - a dataframe with cell_id and maximum coherence
    '''
    coh = []
    cells = []
    if method == 'static':
        for cell_id in tqdm(spikes['cell_id'].unique()):
            ts = spikes.query('cell_id==@cell_id')['ts'].values
            cc = get_coherence(ts,x,x_sr,t0,tf)[0]
            coh.append(cc)
            cells.append(cell_id)
    elif method == 'moving':
        # use this method to break the coherence calculations up into chunks because breathing changes rates.
        # probably do not use
        raise Warning('This method is not advised as of 2021-06-10. It nans out sometimes, and causes inflation of low coherence cells')
        binsize=100
        t_bins = np.arange(t0,tf,binsize)
        for cell_id in tqdm(spikes['cell_id'].unique()):
            ts = spikes.query('cell_id==@cell_id')['ts'].values
            ts = spikes.query('cell_id==@cell_id')['ts'].values
            temp = []
            for bin in t_bins[:-1]:
                cc,sfc,freqs = get_coherence(ts,x,x_sr,t0=bin,tf=(bin+binsize))
                temp.append(cc)
            temp = np.array(temp)
            temp = np.mean(temp)
            coh.append(temp)
            cells.append(cell_id)

        coh_df = pd.DataFrame()
        coh_df['cell_id'] = cells
        coh_df['coherence'] = coh




def event_average_mod_depth(spikes,events,pre=0.25,post=0.5,method='sqrt'):
    '''
    Computes the modulation depth of the average firing rate with respect to a given event
    The modulation depth is defined as: (maxFR-min_FR)/sqrt(maxFR+minFR)
    This has not been thoroughly validated as a good metric, but it allows for decent scaling properties it seems

    returns a dataframe with cell id and modulation depth as columns. cell_id will index appropriately with the cell_id
    from the "spikes" dataframe

    :param spikes: the spikes dataframe
    :param events: the events to time lock to and average
    :param pre: [seconds] the window prior to the event to consider (default =0.25)
    :param post: [seconds] the window after the event to consider (default=0.5)
    :return: dataframe of cell_id vs modulation depth
    '''
    raster,cell_id,bins = bin_trains(spikes.ts,spikes.cell_id,max_time=events[-1],binsize=.015)
    TT_eup,raster_bins = raster2tensor(raster,bins,events,pre=pre,post=post)
    raster_bins = raster_bins[1:]-(raster_bins[1]-raster_bins[0])/2
    mean_eup = np.mean(TT_eup,2)
    if method=='sqrt':
        mod_depth = np.abs((np.max(mean_eup, 0) - np.min(mean_eup, 0)) / np.sqrt((np.max(mean_eup, 0) + np.min(mean_eup, 0))))
    else:
        mod_depth = np.abs((np.max(mean_eup, 0) - np.min(mean_eup, 0)) /((np.max(mean_eup, 0) + np.min(mean_eup, 0))))
    df = pd.DataFrame()
    df['cell_id'] = cell_id
    df['event_triggered_modulation'] = mod_depth
    return(df)


def get_binned_event_rate(evt,dt,start_time=0,stop_time=None,method='hist'):
    '''

    Use this as an alternative to the instantaneous frequency.
    Typical use is to get the breathing rate.

    :param evt: list or array of events (units of time - typically seconds)
    :param dt: window size to count the number of events over. MUST BE SAME UNITS AS evt
    :param start_time: must be same units as evt
    :param stop_time: default=None; if None, takes the last event as the end time
    :param method: hist, gaussian, boxcar. Hist uses non overlapping windows, gaussian and boxcar use sliding windows
    :return: t_vec,rate
    '''

    # Use the last event as the stop time if none given
    if stop_time is None:
        stop_time = np.max(evt)+0.1 #this epsilon allows for last event to be included

    # Make sure we only use events before the stop time
    last_event = np.searchsorted(evt,stop_time)
    evt = evt[:last_event]

    # Hist method is very coarse; uses non overlapping bins to get the number of events in
    # equally spaced time windows
    if method=='hist':
        bins = np.arange(start_time, stop_time, dt)
        rate,t_vec = np.histogram(evt,bins)
        t_vec = t_vec[:-1]
        rate  = rate/dt
    # Gaussian  passes the event as a boolean to a smoothing convolution
    elif method=='gaussian':
        res = 0.1
        t_vec = np.arange(start_time,stop_time,res)
        idx = np.searchsorted(t_vec,evt)
        b_evt = np.zeros_like(t_vec)
        b_evt[idx]=1
        rate = scipy.ndimage.gaussian_filter1d(b_evt,sigma=dt/res)
        rate =rate/res
    # other methods try to pass "method" as a window" for convolution
    else:
        res = 0.1
        t_vec = np.arange(start_time,stop_time,res)
        idx = np.searchsorted(t_vec,evt)
        b_evt = np.zeros_like(t_vec)
        b_evt[idx]=1
        try:
            kernel = scipy.signal.get_window(method,int(dt/res))
        except:
            raise NotImplementedError(f'Chosen method: {method} is not supported. Choose ["hist","gaussian"] or a scipy.signal.get_window() supported window')
        rate = scipy.ndimage.convolve1d(b_evt,kernel)
        rate =rate/np.sum(kernel)/res
    return(t_vec,rate)


def raster2tensor(raster,raster_bins,events,pre = .100,post = .200):
    '''
    Given the binned spikerates over time and a series of events,
    creates a tensor that is shape [n_bins_per_trial,n_cells,n_trials]

    :param raster: a [time x neurons] array of spike rates
    :param raster_bins: array of times in seconds for each bin ( first dim of raster)
    :param events: array of times in seconds at which each event started
    :param pre: float, amount of time prior to event onset to consider a trial(positive, seconds)
    :param post: float, amount of time after event onset to consider a trial (positive, seconds)
    :return:
        raster_T    - [time x cells x trials] tensor of spike rates for each trial (event)
        bins        - array of values in seconds that describes the first dimension of raster_T
    '''
    dt = np.round(np.mean(np.diff(raster_bins)),5)
    trial_length = int(np.round((pre+post)/dt))
    keep_events = events>(raster_bins[0]-pre)
    events = events[keep_events]
    keep_events = events<(raster_bins[-1]-post)
    events = events[keep_events]

    raster_T = np.empty([trial_length,raster.shape[0],len(events)])

    for ii,evt in enumerate(events):
        t0 = evt-pre
        t1 = evt+post
        bin_lims = np.searchsorted(raster_bins,[t0,t1])
        xx = raster[:,bin_lims[0]:bin_lims[0]+trial_length].T
        raster_T[:,:,ii]= xx
    bins = np.arange(-pre,post,dt)
    return(raster_T,bins)





