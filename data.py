"""Routines for data import and manipulation."""
import sys
import csv
import glob
import pandas as pd
import os
import numpy as np
from . import readSGLX as readSGLX
from pathlib import Path
from utils.ephys.signal import binary_onsets
import utils.burst as burst

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








