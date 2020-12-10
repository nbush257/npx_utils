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


def create_spike_dict(ks2_dir,clus_id=None):
    '''
    load in the spike times from an npy file and seperate into a dict
    where each cluster is a field. Optionally keep only certain clusters


    :param ks2_dir: directory of the kilosort output
    :param clus_id: List of clusters to keep (default keeps all)
    :return: spiketimes dict
    '''
    st_fn = os.path.join(ks2_dir,'spike_times_sec.npy')
    sc_fn = os.path.join(ks2_dir,'spike_clusters.npy')
    amp_fn = os.path.join(ks2_dir,'amplitudes.npy')
    # ks_label = pd.read_csv(os.path.join(ks2_dir,'cluster_KSLabel.tsv'),delimiter='\t')
    # group = pd.read_csv(os.path.join(ks2_dir,'cluster_group.tsv'),delimiter='\t')

    st = np.load(st_fn)
    sc = np.load(sc_fn)
    amp = np.load(amp_fn)
    chan_locs = np.load(os.path.join(ks2_dir,'channel_positions.npy'))
    clu_info = pd.read_csv(os.path.join(ks2_dir,'cluster_info.tsv'),delimiter='\t')

    if clus_id is None:
        clus_id = clu_info.id.unique()
    spike_dict = {}
    for clu in clus_id:
        temp = {}
        temp['ts'] = st[sc==clu].ravel()
        temp['amp'] = amp[sc==clu].ravel()

        temp['pk_channel'] = clu_info[clu_info['id']==clu].ch.values[0]
        temp['depth'] = clu_info[clu_info['id']==clu].depth.values[0]
        # temp['depth'] = chan_locs[temp['pk_channel']][1]
        temp['n_spikes'] = len(temp['ts'])
        temp['mean_amp'] = np.mean(temp['amp'])
        temp['group'] = clu_info[clu_info['id']==clu].group.values[0]
        spike_dict[clu] = temp

    return(spike_dict)


def create_spike_df(ks2_dir):
    '''
    Loads in the spike time data to create a dataframe
    Filters out only to the "good" cells
    :param ks2_dir:
    :return: spike_df - pandas dataframe of cluster data
    '''
    spike_df = pd.DataFrame(create_spike_dict(ks2_dir)).T
    spike_df = spike_df[spike_df.group=='good']
    spike_df.drop('amp',axis=1,inplace=True)
    spike_df.reset_index(inplace=True)
    spike_df.rename({'index':'clu_id'},axis=1)

    return(spike_df)


def get_concatenated_spikes(ks2_dir):
    '''
    Built on top of create_spike_dict and create_spike_df
    Returns a minimal set of concatenated spike data. Probably the most useful
    way to import spike data
    :param ks2_dir:
    :type ks2_dir:
    :return:
    :rtype:
    '''

    spike_df = create_spike_df(ks2_dir)


    sp = []
    sc = []
    depth = []
    for ii,data in spike_df.iterrows():
        sp_temp = data['ts']
        sc_temp = np.repeat(ii,len(sp_temp))
        depth_temp = np.repeat(data['depth'],len(sp_temp))

        sp.append(sp_temp)
        sc.append(sc_temp)
        depth.append(depth_temp)
    sp = np.hstack(sp)
    sc = np.hstack(sc)
    depth = np.hstack(depth)

    idx = np.argsort(sp)
    sp = sp[idx]
    sc = sc[idx].astype('int')
    depth = depth[idx]

    cat_spike_dict = pd.DataFrame()
    cat_spike_dict['ts'] = sp
    cat_spike_dict['cell_id'] = sc
    cat_spike_dict['depth'] = depth
    return(cat_spike_dict)


def filter_by_spikerate(spikes,thresh = 100):
    '''
    remove units that have less than "thresh" spikes in the recording

    :return: spikes2
    '''
    n_spikes = spikes.groupby('cell_id').count()['ts']
    mask = n_spikes[n_spikes>thresh].index
    spikes2 = spikes.loc[spikes['cell_id'].isin(mask)]
    return(spikes2)


def export_goodcell_csv(ks2_dir,ni_bin_fn,ecell_chan=2):
    '''
    Takes sortred KS2 data and exports only good cells to a XLSX
    file for analyses.
    Automatically grabs the extracellular data and finds burst onsets

    :param ks2_dir: Directory of sorted data
    :param ni_bin_fn: Filename for auziliary data
    :param ecell_chan: Channel the extracellular (burst) data is on
    :return: None - saves a summary excel file
    '''
    output_name = os.path.join(ks2_dir,'good_spiketimes.xlsx')

    # Format spike data
    sd = create_spike_dict(ks2_dir)
    df = pd.DataFrame(sd).T
    goods = df['group']=='good'
    df_good = df[goods]
    ts = pd.DataFrame(df_good['ts'].tolist()).T

    sum_dat = df_good.drop(['ts','amp','group','pk_channel'],axis=1)
    sum_dat.reset_index().rename(columns={'index':'original_clu_id'})

    # Format extracellular
    ecell = get_ni_analog(ni_bin_fn,ecell_chan)
    meta = readSGLX.readMeta(Path(ni_bin_fn))
    sr = readSGLX.SampRate(meta)
    # x_sync = glob.glob(os.path.join(ks2_dir,'../*XA_0_0*.txt'))
    # sync_timestamps = glob.glob(os.path.join(ks2_dir,'../*SY*.txt'))
    # tvec = get_tvec(x_sync,sync_timestamps,sr)
    pk_df = burst.get_burst_stats(ecell,sr,rel_height=0.85,thresh=1.5)


    ###
    writer = pd.ExcelWriter(output_name,engine='xlsxwriter')
    ts.to_excel(writer,sheet_name='Spike Times (s)',index=False)
    sum_dat.to_excel(writer,sheet_name='Summary Data')
    pk_df.to_excel(writer,sheet_name='Burst Statistics')
    writer.save()


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


def create_spike_mat(spikes,dt = 0.001):
    n_neurons = np.max(spikes.cell_id.unique())+1
    max_t = np.max(spikes.ts)
    t_vec = np.arange(0,max_t,dt)
    X = np.zeros([n_neurons,len(t_vec)+1],dtype='bool')
    for n in range(n_neurons):
        sub_spikes = spikes[spikes.cell_id==n]
        ts = sub_spikes.ts.values
        idx = np.searchsorted(t_vec,ts)
        X[n,idx] = 1

    return(X)


