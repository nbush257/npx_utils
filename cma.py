'''

ported from the matlab code: https: // github.com / kerstinlenk / NetworkWideAdaptiveBurstDetection / blob / master / DetectBurstNoRound.m
'''
import numpy as np
from scipy.stats import skew


def cma_detect_burst(ts, thresh=0.01, tail=1):
    '''
    :param ts: timestamps of a neuron (in ms!)
    :param thresh: spikes within a burst must be closer together than this number to be considered a burts
    :param tail:  not sure what this does yet NEB 20201005
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
    Not finished
    :param ts:
    :type ts:
    :return:
    :rtype:
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
    :param isi: vector of ISI's
    :return: info: a dictionary of needed statistics
    '''
    info = {}
    info['maxISI'] = np.max(isi)
    info['minISI'] = np.min(isi)
    info['meanISI'] = np.mean(isi)
    info['stdISI'] = np.std(isi)
    info['medianISI'] = np.median(isi)
    info['histISI'],bins = np.histogram(isi,20000)
    info['cumhistISI'] = np.cumsum(info['histISI'])
    info['CMAcurve'] = info['cumhistISI'] / bins[:-1]

    if info['maxISI']>20000 or np.isnan(info['maxISI']):
        info['skewCMA'] = skew(info['CMAcurve'])
    else:
        dum = np.ceil(info['maxISI']).astype('int')
        info['skewCMA'] = skew(info['CMAcurve'][:dum])
    info['skewISI'] = skew(isi)
    return(info)


def skw2alpha(skw):
    '''
    Needs testind
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
    Needs testing

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







