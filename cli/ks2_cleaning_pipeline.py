import click
import pandas as pd
import numpy as np
import os
from argschema import ArgSchemaParser
import sys
import subprocess
import platform
sys.path.append(os.environ['PROJ'])
sys.path.append(os.path.join(os.environ['PROJ'],'../helpers'))
from ecephys_spike_sorting.ecephys_spike_sorting.modules.kilosort_helper._schemas import Kilosort2Parameters

from ecephys_spike_sorting.ecephys_spike_sorting.common.utils import rms,load_kilosort_data,write_cluster_group_tsv
from ecephys_spike_sorting.ecephys_spike_sorting.modules.kilosort_helper.SGLXMetaToCoords import MetaToCoords,readMeta,ChannelCountsIM
from ecephys_spike_sorting.ecephys_spike_sorting.modules.kilosort_helper import matlab_file_generator
from scipy.signal import butter,filtfilt,medfilt
from pathlib import Path
from ecephys_spike_sorting.ecephys_spike_sorting.common.utils import load_kilosort_data, getSortResults
from ecephys_spike_sorting.ecephys_spike_sorting.modules.kilosort_postprocessing.postprocessing import remove_double_counted_spikes
from ecephys_spike_sorting.ecephys_spike_sorting.modules.noise_templates.id_noise_templates import id_noise_templates,id_noise_templates_rf
from ecephys_spike_sorting.ecephys_spike_sorting.modules.mean_waveforms.extract_waveforms import extract_waveforms,writeDataAsNpy
from ecephys_spike_sorting.ecephys_spike_sorting.modules.mean_waveforms.metrics_from_file import metrics_from_file
from ecephys_spike_sorting.ecephys_spike_sorting.modules.quality_metrics.metrics import calculate_metrics

"""
First pass attempt to implement the ecephys pipeline on our data.
This is very ugly at moment, but could probably be cleaned up.
Likely useful to properly implement Argschema
"""

def get_noiseWaveformParams():
    '''
    Returns
    -------
    Dictionary of default parameters for the removal of noise waveforms

    '''
    params = {}
    params['smoothed_template_amplitude_threshold'] = 0.2
    params['template_amplitude_threshold'] = 0.2
    params['smoothed_template_filter_width'] = 2
    params['min_spread_threshold'] = 2
    params['mid_spread_threshold'] = 16
    params['max_spread_threshold'] = 25

    params['channel_amplitude_thresh'] = 0.25
    params['peak_height_thresh'] = 0.2
    params['peak_prominence_thresh'] = 0.2
    params['peak_channel_range'] = 24
    params['peak_locs_std_thresh'] = 3.5

    params['min_temporal_peak_location'] = 10
    params['max_temporal_peak_location'] = 30

    params['template_shape_channel_range'] = 12
    params['wavelet_index'] = 2
    params['min_wavelet_peak_height'] = 0.0
    params['min_wavelet_peak_loc'] = 15
    params['max_wavelet_peak_loc'] = 25

    params['multiprocessing_worker_count'] = 4
    params['use_random_forest'] = False
    return params


def makeMemMapRaw(binFullPath, meta):
    #TODO: This is a copy of the function in the readSGLX.py file. Need to do a proper import of this function
    nChan = int(meta['nSavedChans'])
    nFileSamp = int(int(meta['fileSizeBytes'])/(2*nChan))
    print("nChan: %d, nFileSamp: %d" % (nChan, nFileSamp))
    rawData = np.memmap(binFullPath, dtype='int16', mode='r',
                        shape=(nChan, nFileSamp), offset=0, order='F')
    return(rawData)


def get_noise_channels(raw_data_file, meta, num_channels, sample_rate, bit_volts, noise_threshold=20):
    '''
    Analyzes the RMS of each channel and marks super noisy channels to be ignored
    Parameters
    ----------
    raw_data_file : raw imec bin filename
    meta : imported SGLX metadata
    num_channels :
    sample_rate : in Hz
    bit_volts :
    noise_threshold :

    Returns
    -------
    mask : Boolean array of channels to keep

    '''
    noise_delay = 5  # in seconds
    noise_interval = 10  # in seconds

    data = makeMemMapRaw(raw_data_file,meta).T
    num_samples = data.shape[0]


    start_index = int(noise_delay * sample_rate)
    end_index = int((noise_delay + noise_interval) * sample_rate)

    if end_index > num_samples:
        print('noise interval larger than total number of samples')
        end_index = num_samples

    b, a = butter(3, [10 / (sample_rate / 2), 10000 / (sample_rate / 2)], btype='band')

    D = data[start_index:end_index, :] * bit_volts

    D_filt = np.zeros(D.shape)

    for i in range(D.shape[1]):
        D_filt[:, i] = filtfilt(b, a, D[:, i])

    rms_values = np.apply_along_axis(rms, axis=0, arr=D_filt)

    above_median = rms_values - medfilt(rms_values, 11)

    print('number of noise channels: ' + repr(sum(above_median > noise_threshold)))

    return above_median < noise_threshold


def Chan0_uVPerBit(meta):
    # Returns uVPerBit conversion factor for channel 0
    # If all channels have the same gain (usually set that way for
    # 3A and NP1 probes; always true for NP2 probes), can use
    # this value for all channels.

    imroList = meta['imroTbl'].split(sep=')')
    # One entry for each channel plus header entry,
    # plus a final empty entry following the last ')'
    # channel zero is the 2nd element in the list

    if 'imDatPrb_dock' in meta:
        # NP 2.0; APGain = 80 for all channels
        # voltage range = 1V
        # 14 bit ADC
        uVPerBit = (1e6)*(1.0/80)/pow(2,14)
    else:
        # 3A, 3B1, 3B2 (NP 1.0)
        # voltage range = 1.2V
        # 10 bit ADC
        currList = imroList[1].split(sep=' ')   # 2nd element in list, skipping header
        APgain = float(currList[3])
        uVPerBit = (1e6)*(1.2/APgain)/pow(2,10)

    return(uVPerBit)

def get_ephys_params(meta):
    '''

    Parameters
    ----------
    meta : metadata dict from SGLX file

    Returns
    -------
    dict of parameters that define the ephys metadata

    '''
    ephys_params = {}
    ephys_params['sample_rate'] = float(meta['imSampRate'])
    ephys_params['bit_volts'] = Chan0_uVPerBit(meta)
    ephys_params['num_channels'] = ChannelCountsIM(meta)[0]
    ephys_params['template_zero_padding'] = True
    ephys_params['vertical_site_spacing'] = 20e-6

    return(ephys_params)


def run_ks2_post(ks2_output_dir,sample_rate):
    '''
    Postprocesses kilsort output by removing double counted spikes

    Parameters
    ----------
    ks2_output_dir : Directory to KS2 results
    sample_rate :

    Returns
    -------
    None - modifies KS2 result files in place

    '''

    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, \
    channel_pos, clusterIDs, cluster_quality, cluster_amplitude, pc_features, pc_feature_ind, template_features = \
        load_kilosort_data(ks2_output_dir, \
                           sample_rate, \
                           convert_to_seconds = False,
                           use_master_clock = False,
                           include_pcs = True)

    ks_post_params = {}
    ks_post_params['within_unit_overlap_window'] = 0.000166
    ks_post_params['between_unit_overlap_window'] = 0.000166
    ks_post_params['between_unit_dist_um'] = 5
    ks_post_params['deletion_mode'] = 'lowAmpCluster'


    spike_times, spike_clusters, spike_templates, amplitudes, pc_features, \
    template_features, overlap_matrix, overlap_summary = \
        remove_double_counted_spikes(spike_times,
                                     spike_clusters,
                                     spike_templates,
                                     amplitudes,
                                     channel_map,
                                     channel_pos,
                                     templates,
                                     pc_features,
                                     pc_feature_ind,
                                     template_features,
                                     cluster_amplitude,
                                     sample_rate,
                                     ks_post_params)

    print("Saving data...")

    # save data -- it's fine to overwrite existing files, because the original outputs are stored in rez.mat
    output_dir = ks2_output_dir
    np.save(os.path.join(output_dir, 'spike_times.npy'), spike_times)
    np.save(os.path.join(output_dir, 'amplitudes.npy'), amplitudes)
    np.save(os.path.join(output_dir, 'spike_clusters.npy'), spike_clusters)
    np.save(os.path.join(output_dir, 'spike_templates.npy'), spike_templates)
    np.save(os.path.join(output_dir, 'pc_features.npy'), pc_features)
    np.save(os.path.join(output_dir, 'template_features.npy'), template_features)
    np.save(os.path.join(output_dir, 'overlap_matrix.npy'), overlap_matrix)
    np.save(os.path.join(output_dir, 'overlap_summary.npy'), overlap_summary)

    # save the overlap_summary as a text file -- allows user to easily understand what happened
    np.savetxt(os.path.join(output_dir, 'overlap_summary.csv'), overlap_summary, fmt = '%d', delimiter = ',')

    # remoake the clus_Table.npy with the new spike counts
    getSortResults(output_dir)
    return('Finished Postprocessing!')


def noise_templates(ks2_output_dir,sample_rate):
    '''
    Remove clusters and waveforms that are clearly noise and write
    that identity to a TSV
    Parameters
    ----------
    ks2_output_dir : Directory to KS2 results
    sample_rate : AP data sample rate in Hz

    Returns
    -------
    None - writes a TSV to ks2_output_dir

    '''
    print('ecephys spike sorting: noise templates module')


    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, \
    channel_pos, cluster_ids, cluster_quality, cluster_amplitude = \
        load_kilosort_data(ks2_output_dir, \
                           sample_rate=sample_rate, \
                           convert_to_seconds=True)

    params = get_noiseWaveformParams()
    params['classifier_path'] = ks2_output_dir

    if params['use_random_forest']:
        # use random forest classifier
        cluster_ids, is_noise = id_noise_templates_rf(spike_times, spike_clusters, \
                                                      cluster_ids, templates, params)
    else:
        # use heuristics to identify templates that look like noise
        cluster_ids, is_noise = id_noise_templates(cluster_ids, templates, np.squeeze(channel_map), \
                                                   params)

    mapping = {False: 'good', True: 'noise'}
    labels = [mapping[value] for value in is_noise]

    cluster_group_fn = 'cluster_group.tsv'
    write_cluster_group_tsv(cluster_ids,
                            labels,
                            ks2_output_dir,
                            cluster_group_fn)


def get_mean_waveforms_param():
    '''

    Returns
    -------
    default parameter dictionary for the
    get mean waveforms processing

    '''
    params = {}
    params['samples_per_spike'] = 82
    params['pre_samples'] = 20
    params['num_epochs'] = 1
    params['spikes_per_epoch'] = 100
    params['upsampling_factor'] = 200/82
    params['spread_threshold'] = 0.12
    params['site_range'] = 16
    params['use_C_Waves'] = False
    params['snr_radius'] = 8
    return(params)


def mean_waveforms_cwaves(ap_fn,ks2_output_dir):
    '''
    Use C_waves to calculate the mean spike waveform shapes
    for each cluster
    DO NOT USE ON LINUX
    Parameters
    ----------
    ap_fn: filename to the SGLX ap bin data
    ks2_output_dir : Directory to KS2 results

    Returns
    -------
    1 - Runs C_waves and writes waveform metrics files

    '''
    cwaves_path = os.path.join('Y:/helpers/C_Waves')
    clus_table_npy = os.path.join(ks2_output_dir,'clus_Table.npy')
    clus_time_npy = os.path.join(ks2_output_dir,'spike_times.npy')
    clus_lbl_npy = os.path.join(ks2_output_dir,'spike_clusters.npy')
    dest = ks2_output_dir
    exe_path = os.path.join(cwaves_path,'C_Waves.exe')
    mwvf_params = get_mean_waveforms_param()
    meta_fn = os.path.splitext(ap_fn)[0] + '.meta'
    meta = readMeta(Path(meta_fn))
    data = makeMemMapRaw(ap_fn,meta).T
    site_spacing = 20 # Hardcoded to 20uM
    bit_volts = Chan0_uVPerBit(meta)
    sample_rate = float(meta['imSampRate'])
    metrics_file = os.path.join(ks2_output_dir,'waveform_metrics.csv')

    cwaves_cmd = exe_path + ' -spikeglx_bin=' + ap_fn + \
                 ' -clus_table_npy=' + clus_table_npy + \
                 ' -clus_time_npy=' + clus_time_npy + \
                 ' -clus_lbl_npy=' + clus_lbl_npy + \
                 ' -dest=' + dest + \
                 ' -samples_per_spike=' + repr(mwvf_params['samples_per_spike']) + \
                 ' -pre_samples=' + repr(mwvf_params['pre_samples']) + \
                 ' -num_spikes=' + repr(mwvf_params['spikes_per_epoch']) + \
                 ' -snr_radius=' + repr(mwvf_params['snr_radius'])

    subprocess.call(cwaves_cmd)


    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, \
    channel_pos, clusterIDs, cluster_quality, cluster_amplitude = \
        load_kilosort_data(ks2_output_dir, \
                           sample_rate, \
                           convert_to_seconds=False)

    mean_waveform_fullpath = os.path.join(dest, 'mean_waveforms.npy')
    snr_fullpath = os.path.join(dest, 'cluster_snr.npy')

    metrics = metrics_from_file(mean_waveform_fullpath, snr_fullpath, \
                                spike_times, \
                                spike_clusters, \
                                templates, \
                                channel_map, \
                                bit_volts, \
                                sample_rate, \
                                site_spacing, \
                                mwvf_params)

    metrics.to_csv(metrics_file)
    return(1)


def mean_waveforms(ap_fn,ks2_output_dir):
    '''
    Use python to calculate the mean spike waveform shapes
    for each cluster

    For use un Linux
    Parameters
    ----------
    ap_fn: filename to the SGLX ap bin data
    ks2_output_dir : Directory to KS2 results

    Returns
    -------
    1 - Runs C_waves and writes waveform metrics files

    '''

    meta_fn = os.path.splitext(ap_fn)[0] + '.meta'
    meta = readMeta(Path(meta_fn))
    data = makeMemMapRaw(ap_fn,meta).T
    site_spacing = 20 # Hardcoded to 20uM
    bit_volts = Chan0_uVPerBit(meta)
    mwvf_params = get_mean_waveforms_param()
    sample_rate = float(meta['imSampRate'])
    metrics_file = os.path.join(ks2_output_dir,'waveform_metrics.csv')

    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, \
    channel_pos, clusterIDs, cluster_quality, cluster_amplitude = \
        load_kilosort_data(ks2_output_dir, \
                           sample_rate, \
                           convert_to_seconds=False)

    print("Calculating mean waveforms...")

    waveforms, spike_counts, coords, labels, metrics = extract_waveforms(data, spike_times, \
                                                                         spike_clusters,
                                                                         templates,
                                                                         channel_map,
                                                                         bit_volts, \
                                                                         sample_rate, \
                                                                         site_spacing, \
                                                                         mwvf_params)

    writeDataAsNpy(waveforms, os.path.join(ks2_output_dir,'mean_waveforms.npy'))
    metrics.to_csv(metrics_file)


def get_qc_params():
    '''

    Returns
    -------
    Dict of default parameters for quality control processing

    '''
    qc_params = {}
    qc_params['isi_threshold'] = 0.0015
    qc_params['min_isi'] = 0.00
    qc_params['num_channels_to_compare'] = 13
    qc_params['max_spikes_for_unit'] = 500
    qc_params['max_spikes_for_nn'] = 10000
    qc_params['n_neighbors'] = 4
    qc_params['n_silhouette'] = 10000

    qc_params['drift_metrics_min_spikes_per_interval'] = 10
    qc_params['drift_metrics_interval_s'] = 100

    return(qc_params)


def QC(ks2_output_dir,ap_fn):
    '''
    Run quality control metric calculation
    Parameters
    ----------
    ks2_output_dir : Directory to KS2 results
    ap_fn: filename to the SGLX ap bin data

    Returns
    -------
    Writes cluster_metrics.csv file and appends to waveform_metrics.csv file

    '''

    qc_params = get_qc_params()
    meta_fn = os.path.splitext(ap_fn)[0] + '.meta'
    meta = readMeta(Path(meta_fn))
    num_channels = ChannelCountsIM(meta)[0]
    sample_rate = float(meta['imSampRate'])
    bit_volts = Chan0_uVPerBit(meta)

    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, \
    channel_pos, clusterIDs, cluster_quality, cluster_amplitude, pc_features, pc_feature_ind, template_features = \
        load_kilosort_data(ks2_output_dir, \
                           sample_rate, \
                           use_master_clock=False,
                           include_pcs=True)

    metrics = calculate_metrics(spike_times, spike_clusters, amplitudes, channel_map, channel_pos, pc_features,
                                pc_feature_ind, qc_params)

    if os.path.exists(os.path.join(ks2_output_dir,'waveform_metrics.csv')):
        metrics = metrics.merge(pd.read_csv(os.path.join(ks2_output_dir,'waveform_metrics.csv')),
                                on='cluster_id',
                                suffixes=('_quality_metrics','_waveform_metrics'))
    output_file = os.path.join(ks2_output_dir,'cluster_metrics.csv')
    metrics.to_csv(output_file)



@click.command()
@click.argument('ks2_output_dir')
@click.argument('ap_fn')
def main(ks2_output_dir,ap_fn):
    '''
    CLI to run postprocessing on a kilosort2 finished sort.
    Currently implemented to run on cluster.
    After this, need to run TPrime
    '''

    meta_fn = os.path.splitext(ap_fn)[0] + '.meta'
    meta = readMeta(Path(meta_fn))
    sample_rate = float(meta['imSampRate'])

    print('Running Kilosort postprocessing')
    run_ks2_post(ks2_output_dir,sample_rate)
    print('Removing noise clusters')
    noise_templates(ks2_output_dir,sample_rate)
    print('Calculating Mean waveforms')
    mean_waveforms(ap_fn,ks2_output_dir)
    print('Calculating QC metrics')
    QC(ks2_output_dir,ap_fn)

if __name__=='__main__':
    main()
