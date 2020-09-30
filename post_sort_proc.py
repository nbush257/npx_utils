from scipy.signal import hilbert,savgol_filter,find_peaks
from tqdm import tqdm
import scipy.signal
import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture as GM
from sklearn.preprocessing import MinMaxScaler



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


def calc_is_bursting(spiketimes,thresh=0.05):
    '''
    :param spiketimes: vector of spike times for a single neuron
    :param thresh: Seperation of means (in seconds) required to classify the isi histograms as bimodal (default=10ms)
    :return: is_burst, clf: Boolean is a burster or not, clf the GMM that gave rise to that result
    :rtype:
    '''
    if len(spiketimes)<200:
        return(False,None)
    isi = np.diff(spiketimes)
    clf = GM(2)
    clf.fit_predict(np.log(isi[:,np.newaxis]))
    exp_means = np.exp(clf.means_)
    mean_diff = exp_means[0]-exp_means[1]
    if np.abs(mean_diff)>thresh:
        is_burst = True
    else:
        is_burst = False
    return(is_burst,clf)


def bin_spiketrains(ts,idx,binsize=.05,start_time=0,max_time=None):
    cell_ids = np.arange(len(np.unique(idx)))
    if max_time is None:
        max_time = np.max(ts)

    bins = np.arange(start_time, max_time, binsize)
    raster = np.empty([len(cell_ids), len(bins)])
    # Remove spikes that happened before the start time
    idx = idx[ts>start_time]
    ts = ts[ts>start_time]
    # Remove spikes that happened after the max time
    idx = idx[ts<max_time]
    ts = ts[ts<max_time]

    # Loop through cells
    for cell in cell_ids:
        cell_ts = ts[idx==cell]
        raster[cell, :-1]= np.histogram(cell_ts, bins)[0]

    return(raster,cell_ids,bins)


def find_all_bursters(ts,idx,thresh=0.05):

    is_burster = np.zeros(np.max(idx)+1,dtype='bool')
    for cell in np.unique(idx):
        is_burster[cell] = calc_is_bursting(ts[idx==cell],thresh=thresh)[0]



def calc_corr_mat(raster,fill_val=0):
    '''
    Calculate the cross correlation of all spike trains, allowing for lagged peaks
    :param raster: matrix of spike rates [cells x time]
    :param fill_val: value to fill when a cell does not spike
    :return: corr_mat: pairwise cell cross correlation maximums
    '''
    corr_mat = np.zeros([raster.shape[0],raster.shape[0]])

    for ii in tqdm(range(raster.shape[0])):
        for jj in range(raster.shape[0]):
            t1 = raster[ii]
            t2 = raster[jj]
            if np.logical_or(np.sum(t1)==0,np.sum(t2)==0):
                corr_mat[ii,jj]=fill_val
                continue

            norm_val = np.sqrt(np.dot(t1, t1) * np.dot(t2, t2))

            xcorr =np.correlate(t1,t2)/norm_val
            corr_mat[ii,jj] = np.max(xcorr)

    return(corr_mat)

