'''

ported from the matlab code: https: // github.com / kerstinlenk / NetworkWideAdaptiveBurstDetection / blob / master / DetectBurstNoRound.m
'''
import numpy as np
from scipy.stats import skew


def cma_detect_burst(ts, thresh=0.01, tail=1):
    '''
    :param ts: timestamps of a neuron (in ms!)
    :param thresh: burst threshold as computed by the ISI distribution
    :param tail:  tail threshold as computed from the ISI distribution
    :return: D - dictionary of burst information:
        burst_starts
        burst_ends
        burst_durations
        n_spikes_in_burst
        burst_numbers
    '''
    n_spikes = len(ts)
    type_spike = np.zeros(n_spikes, 'int')
    isis = np.where(np.diff(ts) < thresh)[0]
    bspikes = np.zeros(len(ts), dtype='bool')
    bspikes[isis] = 1
    bspikes[isis + 1] = 1
    type_spike[bspikes] = 1
    min_spikes = 3
    break_point_after = np.array([])
    break_point_before = np.array([])
    burst_spikes = np.where(type_spike == 1)[0]

    if len(burst_spikes) >= 1:
        break_points = np.where(np.diff(ts[burst_spikes]) > thresh)[0]
        break_point_before = np.hstack([burst_spikes[0], burst_spikes[break_points + 1]])
        break_point_after = np.hstack([burst_spikes[break_points], burst_spikes[-1]])

    zero_idx = np.where(break_point_after - break_point_before + 1 < min_spikes)[0]

    for ii in zero_idx:
        type_spike[break_point_before[ii]:type_spike[break_point_after[ii]]] = 0

    isis = np.where(np.diff(ts) < tail)[0]
    tspikes = np.zeros(len(ts), 'bool')
    tspikes[isis] = 1
    tspikes[isis + 1] = 1
    tspikes[bspikes] = 0
    type_spike[tspikes] = 2

    tailspikes = np.where(type_spike == 2)[0]
    controlForLoop = type_spike
    resultOfMerging = np.zeros(n_spikes)

    counter = 0
    while np.any(controlForLoop != resultOfMerging):
        counter += 1
        if counter > 100:
            break
        #         print('merging')
        controlForLoop = type_spike
        for t in tailspikes:
            if t == 0:
                cond1 = ts[t + 1] - ts[t] <= tail
                cond2 = type_spike[t + 1] == 1
                if cond1 and cond2:
                    type_spike[t] = 1
            elif t == (n_spikes - 1):
                cond1 = ts[t] - ts[t - 1] <= tail
                cond2 = type_spike[t - 1] == 1
                if cond1 and cond2:
                    type_spike[t] = 1
            else:
                cond1 = type_spike[t + 1] == 1
                cond2 = ts[t + 1] - ts[t] <= tail
                if cond1 and cond2:
                    type_spike[t] = 1
            resultOfMerging = type_spike
            tailspikes = np.where(type_spike == 2)[0]

    nonbursts = type_spike != 1
    type_spike[nonbursts] = 0

    burst_spikes = np.where(type_spike == 1)[0]
    if len(burst_spikes) > 0:
        break_points = np.where(np.diff(ts[burst_spikes]) > tail)[0]
        break_point_starts = np.hstack([burst_spikes[0], burst_spikes[break_points + 1]])
        break_point_ends = np.hstack([burst_spikes[break_points], burst_spikes[-1]])
        burst_starts = ts[break_point_starts]
        burst_ends = ts[break_point_ends]
        n_spikes_in_burst = break_point_ends - break_point_starts
        burst_durations = burst_ends - burst_starts
        burst_numbers = np.zeros(len(type_spike))
        for ii, b in enumerate(break_point_starts):
            burst_numbers[break_point_starts[ii]:break_point_ends[ii]] = ii
    else:
        burst_starts = []
        burst_ends = []
        burst_durations = []
        n_spikes_in_burst = []
        burst_numbers = type_spike
    D = {}
    D['burst_starts'] = burst_starts
    D['burst_ends'] = burst_ends
    D['burst_durations'] = burst_durations
    D['n_spikes_in_burst'] = n_spikes_in_burst
    D['burst_numbers'] = burst_numbers

    return (D)

def calculate_cma_original(ts):
    '''
    :param ts: list of spike timestamps in ms
    :return: bursts dictionary
                burst_starts
                burst_ends
                burst_durations
                n_spikes_in_burst
                burst_numbers
    '''

    isi = np.diff(ts)
    info = isi_info(isi)
    bA,tA = skw2alpha((info['skewISI']))
    bT,tT = alpha2thresh(info['CMAcurve'],tA,bA)
    bursts = cma_detect_burst(ts,bT,tT)

    return(bursts)


def isi_info(isi):
    '''
    Compute metrics from the ISI that are needed for the CMA computation
    :param isi: vector of ISIs in units of ms
    :return: info: a dictionary of needed statistics
    '''
    info = {}
    info['maxISI'] = np.max(isi)
    info['minISI'] = np.min(isi)
    info['meanISI'] = np.mean(isi)
    info['stdISI'] = np.std(isi)
    info['medianISI'] = np.median(isi)
    info['histISI'],bins = np.histogram(isi,np.arange(20001))
    info['cumhistISI'] = np.cumsum(info['histISI'])
    info['CMAcurve'] = info['cumhistISI'] / bins[1:]

    if info['maxISI']>20002 or np.isnan(info['maxISI']):
        info['skewCMA'] = skew(info['CMAcurve'])
    else:
        dum = np.ceil(info['maxISI']).astype('int')
        info['skewCMA'] = skew(info['CMAcurve'][:dum])
    info['skewISI'] = skew(isi)
    return(info)


def skw2alpha(skw):
    '''
    utlity function used in calculation of burstiness
    computes the alpha values given the skew of the ISI distribution
    :param skw:
    :return:
    '''

    if np.isnan(skw):
        return(np.nan,np.nan)
    if skw<1:
        bA = 1
        tA = 0.7
    elif skw<3:
        bA = 0.7
        tA = 0.5
    elif skw<9:
        bA = 0.5
        tA = 0.3
    else:
        bA = 0.3
        tA = 0.1

    return(bA,tA)

def alpha2thresh(cma_curve,bA,tA):
    '''
    utlity function used in calculation of burstiness
    computes the thresholds given the ISI distribution

    :param cma_curve:
    :param bA:
    :param tA:
    :return:
    '''
    maxCMA = np.max(cma_curve)
    maxCMA_idx = np.where(cma_curve==np.max(cma_curve))[0][-1]

    if np.isnan(bA) or len(cma_curve)<1:
        bT = np.nan
    else:
        bD = np.abs(cma_curve[maxCMA_idx:] - bA * np.max(maxCMA))
        bT = maxCMA_idx + np.where(bD == np.min(bD))[0][-1]-1

    if np.isnan(tA) or len(cma_curve)<1:
        tT = np.nan
    else:
        tD = np.abs(cma_curve[bT:] - tA * np.max(maxCMA))
        tT = bT + np.where(tD==np.min(tD))[0][-1]-1
    return(bT,tT)


def get_bursts(ts,mode = 's'):
    '''
    wrapper to the CMA burst algorithm that allows for ins and outs to be in
    seconds or ms
    Adds a metric that is the CV of the burst durations to indicate how regular the bursting is

    :param ts: timestamps of the neuron's spiketimes
    :param mode:  's' for inputs and outputs in seconds
                    or 'ms' for millisecond. default = 's'
    :return: burst dictionary
    '''
    if mode =='s':
        ts_ms = ts*1000
        bursts = calculate_cma_original(ts_ms)
        bursts['burst_starts'] = bursts['burst_starts']/1000
        bursts['burst_ends'] = bursts['burst_ends']/1000
        bursts['burst_durations'] = bursts['burst_durations']/1000
    elif mode == 'ms':
        bursts = calculate_cma_original(ts)
    else:
        raise NotImplemented(f'Mode {mode} must be "s" or "ms"')
    bursts['CV'] = np.std(bursts['burst_durations'])/np.mean(bursts['burst_durations'])
    return(bursts)


def filter_bursts(bursts,min_dur=None,max_dur=None,min_spikes=None,min_postBI=None):
    '''
    new_bursts = filter_bursts(bursts,min_dur=None,max_dur=None,min_spikes=None,min_postBI=None):

    Remove some bursts if they do not meet particular criteria
    :param bursts: input burst dictionary
    :param min_dur: minimum burst duration
    :param max_dur: maximum burst duration
    :param min_spikes: minimum number of spikes in a given burst
    :param min_postBI: minimum amount of time until the next burst
    :return: new_bursts - a copied burst dictionary with the errant bursts filtered out
    '''

    new_bursts = bursts.copy()
    keep = np.ones(len(bursts['burst_durations']),'bool')
    dur = bursts['burst_durations']
    nspikes = bursts['n_spikes_in_burst']
    postBI = np.concatenate([bursts['burst_starts'][1:] - bursts['burst_ends'][:-1],[np.nan]])
    if min_dur is not None:
        keep[dur<min_dur]=False
    if max_dur is not None:
        keep[dur>max_dur]=False
    if min_spikes is not None:
        keep[nspikes<min_spikes]=False
    if min_postBI is not None:
        keep[postBI<min_postBI]=False
    new_bursts['burst_starts'] = bursts['burst_starts'][keep]
    new_bursts['burst_ends'] = bursts['burst_ends'][keep]
    new_bursts['burst_durations'] = bursts['burst_durations'][keep]
    new_bursts['n_spikes_in_burst'] = bursts['n_spikes_in_burst'][keep]

    new_bursts['CV'] = np.std(new_bursts['burst_durations'])/np.mean(new_bursts['burst_durations'])

    # since we have filtered the bursts now, we are removing this metric.
    # I dont know what it is for anyhow
    # If we think it is useful it may take time to test the appropriate removal
    # NEB 20201006

    new_bursts['burst_numbers'] = []

    return(new_bursts)












