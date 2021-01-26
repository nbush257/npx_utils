"""Routines for data import and manipulation."""
import neo
import elephant
import quantities as pq
import sys
import csv
import glob
import pandas as pd
import os
import numpy as np
try:
    from . import readSGLX as readSGLX
except:
    import readSGLX
from pathlib import Path
sys.path.append('../')
sys.path.append('/active/ramirez_j/ramirezlab/nbush/projects')
from utils.ephys.signal import binary_onsets
import utils.burst as burst
import utils.brian_utils.postproc as bup
from spykes.plot import NeuroVis,PopVis
from tqdm import tqdm


def spike_times_npy_to_sec(sp_fullPath, sample_rate = 0, bNPY = True):
    # convert spike_times.npy to text of times in sec
    # return path to the new file. Can take sample_rate as a
    # parameter, or set to 0 to read from param file

    # get file name and create path to new file
    # FROM ECEPHYS, not NEB
    '''
    sp_fullPath: path to kilosort spiketimes output (data are in samples)
    sample_rate: default to 0 in order to read in from the param file
    bNPY: boolean flag for using npy files. Default to True. Vestigial but kept in case we need the flexibility

    '''

    sp_path, sp_fileName = os.path.split(sp_fullPath)
    baseName, bExt = os.path.splitext(sp_fileName)
    if bNPY:
        new_fileName = baseName + '_sec.npy'
    else:
        new_fileName = baseName + '_sec.txt'

    new_fullPath = os.path.join(sp_path, new_fileName)

    # load spike_times.npy; returns numpy array (Nspike,) as uint64
    spike_times = np.load(sp_fullPath)

    if sample_rate == 0:
        # get sample rate from params.py file, assuming sp_path is a full set
        # of phy output
        with open(os.path.join(sp_path, 'params.py'), 'r') as f:
            currLine = f.readline()
            while currLine != '':  # The EOF char is an empty string
                if 'sample_rate' in currLine:
                    sample_rate = float(currLine.split('=')[1])
                    print(f'sample_rate read from params.py: {sample_rate:.10f}')
                currLine = f.readline()

            if sample_rate == 0:
                print('failed to read in sample rate\n')
                sample_rate = 30000

    spike_times_sec = spike_times/sample_rate   # spike_times_sec dtype = float

    if bNPY:
        # write out npy file
        np.save(new_fullPath, spike_times_sec)
    else:
        # write out single column text file
        nSpike = len(spike_times_sec)
        with open(new_fullPath, 'w') as outfile:
            for i in range(0, nSpike-1):
                outfile.write(f'{spike_times_sec[i]:.6f}\n')
            outfile.write(f'{spike_times_sec[nSpike-1]:.6f}')

    return new_fullPath


def get_tvec(x_sync,sync_timestamps,sr):
    '''
    Get a time vector for NI that uses the sync signal

    :param x_sync:
    :param sync_timestamps: should be from the IMEC master probes
    :return: tvec
    '''
    tvec = np.empty_like(x_sync)
    onsets,offsets = binary_onsets(x_sync,1)
    n_ons = len(onsets)
    n_ts = len(sync_timestamps)
    if n_ons!=n_ts:
        raise ValueError(f"number of detected onsets {n_ons} does not match number expected {n_ts}")

    # Map all detected onsets to a timestamp and linearly interpolate
    for ii in range(len(onsets)-1):
        nsamps = onsets[ii+1]-onsets[ii]
        temp = np.linspace(sync_timestamps[ii],sync_timestamps[ii+1],nsamps)
        tvec[onsets[ii]:onsets[ii+1]] = temp

    # get times before first synch signal
    first_t = sync_timestamps[0]-onsets[0]/sr
    first_seg = np.linspace(first_t,sync_timestamps[0],onsets[0])
    tvec[:onsets[0]] = first_seg

    # get times after last synch signal
    last_seg_length = len(x_sync)-onsets[-1]
    last_seg = np.linspace(sync_timestamps[-1],sync_timestamps[-1] + last_seg_length/sr,last_seg_length)
    tvec[onsets[-1]:] = last_seg

    return(tvec)


def get_sr(ni_bin_fn):
    meta = readSGLX.readMeta(Path(ni_bin_fn))
    sr = readSGLX.SampRate(meta)
    return(sr)


def get_ni_analog(ni_bin_fn, chan_id):
    '''
    Convinience function to load in a NI analog channel
    :param ni_bin_fn: filename to load from
    :param chan_id: channel index to load
    :return: analog_dat
    '''
    meta = readSGLX.readMeta(Path(ni_bin_fn))
    bitvolts = readSGLX.Int2Volts(meta)
    ni_dat = readSGLX.makeMemMapRaw(ni_bin_fn,meta)
    analog_dat = ni_dat[chan_id]*bitvolts

    return(analog_dat)


def get_concatenated_spikes(ks2_dir,use_label='default'):
    '''
    Built on top of create_spike_dict and create_spike_df
    Returns a minimal set of concatenated spike data. Probably the most useful
    way to import spike data
    :param ks2_dir:
    :param use_label: define whether to use the phy label(default_, ks_label, or the intersection
    '''

    # spike_df = create_spike_df(ks2_dir)
    ts = np.load(f'{ks2_dir}/spike_times_sec.npy')
    idx = np.load(f'{ks2_dir}/spike_clusters.npy')
    metrics = pd.read_csv(f'{ks2_dir}/metrics.csv',index_col=0)
    depths = np.load(f'{ks2_dir}/channel_positions.npy')[:,1]
    dd = pd.DataFrame()
    dd['peak_channel'] = np.arange(len(depths))
    dd['depth'] = depths
    metrics = metrics.merge(dd,how='left',on='peak_channel')

    spikes = pd.DataFrame()
    spikes['ts'] = ts
    spikes['cell_id'] = idx
    spikes = pd.merge(left=spikes, right=metrics[['cluster_id', 'depth']], how='left', left_on='cell_id',
                      right_on='cluster_id')

    if use_label == 'default':
        grp = pd.read_csv(f'{ks2_dir}/cluster_group.tsv', delimiter='\t')
        clu_list = grp.query('group=="good"')['cluster_id']
        spikes = spikes[spikes['cluster_id'].isin(clu_list)]
        metrics = metrics.merge(grp,on='cluster_id')
    elif use_label == 'ks':
        grp = pd.read_csv(f'{ks2_dir}/cluster_KSLabel.tsv', delimiter='\t')
        clu_list = grp.query('KSLabel=="good"')['cluster_id']
        spikes = spikes[spikes['cluster_id'].isin(clu_list)]
        metrics = metrics.merge(grp,on='cluster_id')
    elif use_label == 'intersect':
        grp = pd.read_csv(f'{ks2_dir}/cluster_group.tsv', delimiter='\t')
        kslabel = pd.read_csv(f'{ks2_dir}/cluster_KSLabel.tsv', delimiter='\t')
        temp = pd.merge(grp,kslabel,how='inner',on='cluster_id')
        metrics = metrics.merge(grp,on='cluster_id')
        temp.query('group=="good" & KSLabel=="good"',inplace=True)
        clu_list = temp['cluster_id']
        spikes = spikes[spikes['cluster_id'].isin(clu_list)]
    else:
        raise NotImplementedError('Use a valid label filter[default,ks,intersect]')


    return(spikes,metrics)


def filter_by_metric(metrics,spikes,expression):
    '''
    Finds all the clusters that pass a particular QC metrics filter expression and keeps only the spikes from
    those clusters
    :param metrics: the metrics csv
    :param spikes: the spikes dataframe with columns [ts,cluster_id,depth]
    :param expression: logical expression to filter the spikes by
    :return: filtered spikes dataframe

    spikes_filt = filter_by_metric(metrics,spikes,'amplitude_cutoff<0.1')
    '''
    clu_list = metrics.query(expression)['cluster_id']
    spikes = spikes[spikes['cluster_id'].isin(clu_list)]
    return(spikes)


def filter_default_metrics(metrics,spikes):
    '''
    Runs filter_by_metric for a few standard metrics.
    Allen metrics : presence ratio >.95, isi_viol<1, amplitude_cutoff < 0.1
    NEB metrics: isi_viol < 2, amplitude_cutoff < 0.2
    I am not using presence ration because I expect some gasp only neurons.
    Amplitude cutoff should still work although I am not sure if with KS2.5 it still makes sense
    isi_violations should be relaxed a little given the bursty nature of these neurons
    :param metrics:
    :param spikes:
    :return: filtered_spikes
    '''
    spikes = filter_by_spikerate(spikes,100)
    spikes = filter_by_metric(metrics,spikes,'isi_viol<2 ')
    spikes = filter_by_metric(metrics,spikes,'amplitude_cutoff<0.2 ')

    return(spikes)


def filter_by_spikerate(spikes,thresh = 100):
    '''
    remove units that have less than "thresh" spikes in the recording

    :return: spikes2
    '''
    n_spikes = spikes.groupby('cell_id').count()['ts']
    mask = n_spikes[n_spikes>thresh].index
    spikes2 = spikes.loc[spikes['cell_id'].isin(mask)]
    return(spikes2)


def create_neo_trains(ks2_dir):
   cat_spikes = get_concatenated_spikes(ks2_dir)
   train_list = []
   max_time = cat_spikes['ts'].max()
   for ii in np.unique(cat_spikes.cell_id):
       ts = cat_spikes[cat_spikes.cell_id==ii]['ts'].values * pq.second
       dum = neo.spiketrain.SpikeTrain(ts,max_time)
       train_list.append(dum)

   return(train_list)


def create_spykes_pop(spikes,start_time=0,stop_time=np.inf):
    '''
    Convert a spikes dataframe to a Spykes neuron list and population object

    :param spikes: dataframe of spike times "ts" in seconds and "cell_id" in long form.
    :param start_time: ignore spikes before this time (s)
    :param stop_time: ignore spikes after this time (s)
    :return: neuron_list,pop
    '''

    sub_spikes = spikes[spikes['ts']>start_time]
    sub_spikes = sub_spikes[sub_spikes['ts']<stop_time]

    neuron_list = []
    for ii, cell_id in enumerate(sub_spikes['cell_id'].unique()):
        sub_df = sub_spikes[sub_spikes['cell_id'] == cell_id]
        if(len(sub_df.ts))<10:
            neuron = []
        else:
            neuron = NeuroVis(sub_df.ts, ii)
        neuron_list.append(neuron)

    pop = PopVis(neuron_list)
    return(neuron_list,pop)


def get_event_triggered_st(ts,events,idx,pre_win,post_win):
    print('Calculating Time D')
    D = ts-events[:,np.newaxis]
    mask = np.logical_or(D<-pre_win,D>post_win)
    D[mask] = np.nan
    pop = []
    print('Working on all neurons')
    for ii in tqdm(np.unique(idx)):
        trains = []
        for jj in range(len(events)):
            sts = D[jj,idx==ii]
            sts = sts[np.isfinite(sts)]
            trains.append(sts)

        pop.append(trains)
    return(pop)

