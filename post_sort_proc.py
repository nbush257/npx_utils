from scipy.signal import hilbert,savgol_filter,find_peaks
try:
    import tensortools as tt
    has_tt=True
except:
    has_tt = False

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



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


def get_event_triggered_st(ts,events,pre_win,post_win):
    D = ts-events[:,np.newaxis]
    mask = np.logical_and(D<-pre_win,D>post_win)
    D[mask] = np.nan



if has_tt:
    def get_best_TCA(TT,plot_tgl=True):
        ''' Fit ensembles of tensor decompositions.
            Returns the best model with the fewest parameters
            '''
        methods = (
            'ncp_hals',  # fits nonnegative tensor decomposition.
        )
        ranks = range(1,15)
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
        for ii in range(1,15):
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

