from scipy.signal import hilbert,savgol_filter,find_peaks
try:
    import tensortools as tt
    has_tt=True
except:
    has_tt = False

import spykes
import scipy.signal
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
import numpy as np
from tqdm import tqdm
from spykes.plot import NeuroVis


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


def raster2tensor(raster,raster_bins,events,pre = .100,post = .200):
    '''


    :param raster:
    :param raster_bins:
    :param events:
    :param pre:
    :param post:
    :return:
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


if has_tt:
    def get_best_TCA(TT,max_rank=15,plot_tgl=True):
        ''' Fit ensembles of tensor decompositions.
            Returns the best model with the fewest parameters
            '''
        methods = (
            'ncp_hals',  # fits nonnegative tensor decomposition.
        )
        ranks = range(1,max_rank+1)
        ensembles = {}
        m = methods[0]
        ensembles[m] = tt.Ensemble(fit_method=m, fit_options=dict(tol=1e-4))
        ensembles[m].fit(TT, ranks=ranks, replicates=10)

        if plot_tgl:
            plot_opts1 = {
                'line_kw': {
                    'color': 'black',
                    'label': 'ncp_hals',
                },
                'scatter_kw': {
                    'color': 'black',
                    'alpha':0.2,
                },
            }
            plot_opts2 = {
                'line_kw': {
                    'color': 'red',
                    'label': 'ncp_hals',
                },
                'scatter_kw': {
                    'color': 'red',
                    'alpha':0.2,
                },
            }
            plt.close('all')
            fig = plt.figure
            ax = tt.plot_similarity(ensembles[m],**plot_opts1)
            axx = ax.twinx()
            tt.plot_objective(ensembles[m],ax=axx,**plot_opts2)
        mm = []
        for ii in range(1,max_rank+1):
            mm.append(np.median(ensembles[m].objectives(ii)))
        mm = 1-np.array(mm)
        best = np.argmax(np.diff(np.diff(mm)))+1
        optimal = ensembles[m].results[ranks[best]]
        dum_obj = 1
        for decomp in optimal:
            if decomp.obj<dum_obj:
                best_decomp = decomp
                dum_obj = decomp.obj

        if plot_tgl:
            axx.vlines(ranks[best],axx.get_ylim()[0],axx.get_ylim()[1],lw=3,ls='--')
        return(best_decomp,[ax,axx])


def calc_is_bursting(spiketimes,thresh=3,max_small_ISI=0.02,max_long_ISI=5.,max_abs_ISI=2):
    '''
    :param spiketimes: vector of spike times for a single neuron
    :param thresh: Seperation of means (in seconds) required to classify the isi histograms as bimodal (default=10ms)
    :return: is_burst, clf: Boolean is a burster or not, clf the GMM that gave rise to that result
    :rtype:
    '''
    if len(spiketimes)<100:
        return(False)
    isi = np.diff(spiketimes)
    nspikes = len(spiketimes)
    hts, bins = np.histogram(np.log(isi), bins=100)

    mode_idx = scipy.signal.argrelmax(np.log(hts), order=10)[0]

    mode_height = hts[mode_idx]
    idx = np.argsort(mode_height)[::-1]
    top_modes = np.exp(bins[mode_idx[idx]])[:2]
    top_modes = np.sort(top_modes)

    # kick out any neurons that are not firing for a while
    max_ISI = np.percentile(np.exp(isi),99)
    if max_ISI>max_abs_ISI:
        return(False)


    # ratio of long ISI mode to short ISI mode. If large, scell is burstier
    if len(top_modes)<2:
        return(False)

    mean_diff = top_modes[1]/top_modes[0]

    if top_modes[0]>max_small_ISI:
        is_burst=False
    elif top_modes[1]>max_long_ISI:
        is_burst=False
    elif mean_diff>thresh:
        is_burst = True
    else:
        is_burst = False
    return(is_burst)


def find_all_bursters(ts,idx,**kwargs):

    is_burster = np.zeros(np.max(idx)+1)
    for cell in np.unique(idx):
        is_burster[cell] = calc_is_bursting(ts[idx==cell],**kwargs)

    return(is_burster)


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