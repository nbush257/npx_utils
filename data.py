"""Routines for data import and manipulation."""
import sys
import numpy as np
import readSGLX
from pathlib import Path
sys.path.append('../')
from utils.ephys.signal import binary_onsets

def get_tvec(x_sync,sync_timestamps,sr):
    '''
    Get a time vector for NI that uses the sync signal

    :param x_sync:
    :param sync_timestamps:
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
    ks_label = pd.read_csv(os.path.join(ks2_dir,'cluster_KSLabel.tsv'),delimiter='\t')

    st = np.load(st_fn)
    sc = np.load(sc_fn)
    amp = np.load(amp_fn)
    chan_locs = np.load(os.path.join(ks2_dir,'channel_positions.npy'))
    metrics = pd.read_csv(os.path.join(ks2_dir,'cluster_metrics.csv'))

    if clus_id is None:
        clus_id = np.sort(np.unique(sc))
    spike_dict = {}
    for clu in clus_id:
        temp = {}
        temp['ts'] = st[sc==clu].ravel()
        temp['amp'] = amp[sc==clu].ravel()
        temp['label'] = ks_label[ks_label['cluster_id']==clu]['KSLabel'].values[0]
        temp['pk_channel'] = metrics[metrics['cluster_id']==clu].peak_channel.values[0]
        temp['depth'] = chan_locs[temp['pk_channel']][1]
        temp['n_spikes'] = len(temp['ts'])
        temp['mean_amp'] = np.mean(temp['amp'])
        spike_dict[clu] = temp

    return(spike_dict)


