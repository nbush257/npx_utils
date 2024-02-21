'''
Functions to work with optogenetic stimulations, and in particular, tags. 
When run from command line will perform salt tag statistics
'''
import numpy as np
import scipy.io.matlab as sio
from pathlib import Path
import pandas as pd
from brainbox import singlecell
import phylib.io.model
import subprocess
import click
import neurodsp.utils
import matplotlib.pyplot as plt


def compute_pre_post_raster(spike_times,spike_clusters,cluster_ids,stim_times,stim_duration = None, window_time = 0.5,bin_size=0.001,mask_dur = 0.002): 
    """Creates the rasters pre and post stimulation time. Optionally sets periods around onset and offset of light to zero (default behavior)

    Args:
        spike_times (_type_): _description_
        spike_clusters (_type_): _description_
        cluster_ids (_type_): _description_
        stim_times (_type_): _description_
        stim_duration (_type_, optional): _description_. Defaults to None.
        window_time (float, optional): _description_. Defaults to 0.5.
        bin_size (float, optional): _description_. Defaults to 0.001.
        mask_dur (float, optional): _description_. Defaults to 0.002.
    """       
    pre_raster,pre_tscale = singlecell.bin_spikes2D(spike_times,
                                            spike_clusters,
                                            cluster_ids,
                                            align_times=stim_times,
                                            pre_time=window_time+mask_dur,
                                            post_time=-mask_dur, # Calculate until 2 ms before the stimulus
                                            bin_size=bin_size,
                                            )
                                            
    post_raster,post_tscale = singlecell.bin_spikes2D(spike_times,
                                        spike_clusters,
                                        cluster_ids,
                                        align_times=stim_times,
                                        pre_time=-mask_dur, # ignore 2ms after stimulus onset to avoid artifactual spikes
                                        post_time=window_time+mask_dur,
                                        bin_size=bin_size,
                                        )
    
    # if stim_duration exists, remove any spikes within - 1 ms and + mask_duration of offset time
    if stim_duration is not None:
        stim_offsets_samp = np.searchsorted(post_tscale,stim_duration)
        post_raster[:,:,stim_offsets_samp-1:stim_offsets_samp+int(mask_dur/bin_size)] = 0

    return(pre_raster,post_raster)


def run_salt(spike_times,spike_clusters,cluster_ids,stim_times,window_time = 0.5,stim_duration= None):
    """    
    Runs the Stimulus Associated Latency Test (SALT - See Kvitsiani 2013) 
    on given units.

    Must pass data to matlab, and does so via saving to a temporary mat file.
    Automatically deletes the temporary mat file.

    Args:
        spike_times (np.array): times in  seconds of every spike 
        spike_clusters (np.array): cluster assignments of every spike
        cluster_ids (np.array): cluster ids to use.
        stim_times (np.array): onset times of the optogenetic stimulus
        window_time (float): duration in seconds pre and post to make the rasters. [Default = 0.5]

    Returns:
        p_stat: Resulting P value for the Stimulus-Associated spike Latency Test.
        I_stat: Test statistic, difference between within baseline and test-to-baseline information distance values. _description_
    
    """    

    pre_raster,post_raster = compute_pre_post_raster(spike_times,
                                                     spike_clusters,
                                                     cluster_ids,
                                                     stim_times,
                                                     window_time = window_time,
                                                     stim_duration = stim_duration,
                                                     bin_size = 0.001,
                                                     mask_dur=0.002)  

    #Sanitize raster
    if np.any(pre_raster<0) | np.any(post_raster<0):
        print('Warning: spike counts less than zero found. Setting to 0')
        pre_raster[pre_raster<0] = 0 
        post_raster[post_raster<0] = 0

    if np.any(pre_raster>1) | np.any(post_raster>1):
        print(f'Warning: Multiple spikes in a single ms bin found (max is {max(np.max(pre_raster),np.max(post_raster))}). Truncating to 1')
        pre_raster[pre_raster>1] = 1
        post_raster[post_raster>1] = 1

    dat_mat = {}
    dat_mat['pre_raster'] = pre_raster
    dat_mat['post_raster'] = post_raster
    dat_mat['cluster_ids'] = cluster_ids
    sio.savemat('.temp_salt.mat',dat_mat)
    
    command = ["matlab","-batch", "python_SALT('.temp_salt.mat');"]
    subprocess.run(command, check=True)
    
    salt_rez = sio.loadmat('.temp_salt.mat')
    # Path('.temp_salt.mat').unlink()
    p_stat = salt_rez['p_stat']
    I_stat = salt_rez['I_stat']

    return p_stat,I_stat



def compute_tagging_summary(spike_times,spike_clusters,cluster_ids,stim_times,window_time = 0.01,bin_size=0.001):
    """Computes the number of stims that a spike was observed and the pre and post stim spike rate

    Args:
        spike_times (_type_): _description_
        spike_clusters (_type_): _description_
        cluster_ids (_type_): _description_
        stim_times (_type_): _description_
        window_time (float, optional): _description_. Defaults to 0.01.
        bin_size (float, optional): _description_. Defaults to 0.001.
    """    
    n_stims = stim_times.shape[0]
    pre_spikecounts,post_spikecounts = compute_pre_post_raster(spike_times,
                                                     spike_clusters,
                                                     cluster_ids,
                                                     stim_times,
                                                     window_time =window_time,
                                                     bin_size=bin_size)
    pre_spikecounts = pre_spikecounts.sum(2)
    post_spikecounts = post_spikecounts.sum(2)

    n_responsive_stims = np.sum(post_spikecounts.astype('bool'),0)
    pre_spikerate = np.sum(pre_spikecounts,0)/n_stims/window_time
    post_spikerate = np.sum(post_spikecounts,0)/n_stims/window_time

    return(n_responsive_stims,pre_spikerate,post_spikerate)
    

def plot_pop_psth(spike_times,spike_clusters,cluster_ids,stim_times,stim_duration=None):
    """
    Plot the population psth aligned to opto onset. Plots vertical lines for onset, 10ms, and offset

    Args:
        spike_times (_type_): _description_
        spike_clusters (_type_): _description_
        cluster_ids (_type_): _description_
        stim_times (_type_): _description_
        stim_duration (_type_, optional): _description_. Defaults to None.
    """    
    peths = singlecell.calculate_peths(spike_times,spike_clusters,cluster_ids,
                                        stim_times,
                                        pre_time=0.01,
                                        post_time=0.1,
                                        bin_size=0.001,
                                        smoothing=0)[0]
    f = plt.figure()
    plt.pcolormesh(peths.tscale,peths.cscale,peths.means)
    plt.axvline(0,color='w',ls=':')
    plt.axvline(0.01,color='silver',ls=':')
    if stim_duration is not None:
        plt.axvline(stim_duration,color='w',ls=':')


def extract_tagging_from_logs(log_df,opto_df,verbose=True):
    """Finds the opto stims that are associated with a logged tagging episode.

    Args:
        log_df (pandas dataframe): Log from the experiment should be an autogenerated tsv
        opto_df (pandas dataframe): Opto stim dataframe extracted from the analog trace. N.B.: must be synchronized 
        verbose (bool, optional): verbose. Defaults to True.

    Raises:
        NotImplementedError: _description_
        ValueError: _description_
    
    Returns (pandas dataframe): Subdataframe from opto_df that only has the tagging data.
    """    

    # Check if synchronized
    if 'on_sec_corrected' not in opto_df.columns:
        print('Warning: optogenetic stimulations do not appear to be synched to the neural data. "on_sec_corrected" was not found')

    # Extract the tagging start and end
    tag_epoch = log_df.query('label == "opto_tagging"')
    tag_starts = tag_epoch['start_time'].values
    tag_ends = tag_epoch['end_time'].values

    # Handle if more than one or less than 1 tagging episodes were found
    if len(tag_starts)>1:
        raise NotImplementedError("More than one tagging episode has not been implemetned yet")
    elif len(tag_starts)==0:
        raise ValueError("No Tagging episodes found.")
    else:
        pass

    # Subset (add a little buffer time so as not to miss first or last stim)
    tag_start = tag_starts[0]-0.5
    tag_end = tag_ends[0]+0.5
    tags = opto_df.query('on_sec>@tag_start & on_sec<@tag_end')

    #Verbose and return
    print(f'Found {tags.shape[0]} tag stimulations with average duration {tags["dur_sec"].mean():0.02}s') if verbose else None
    return(tags)
    


# TODO: Implement functionality on data from multiple recordings
# TODO: Implement plotting
# TODO: Implement Chrmine option (slower optotagging responses)
@click.command()
@click.argument('ks_dir') # Will not work on concatnated data yet.
@click.argument('opto_fn')
@click.argument('log_fn') # Pass the files explicitly. Folder structure may evolve to be more crystalized
def main(ks_dir,opto_fn,log_fn):
    ks_dir = Path(ks_dir)

    # Load spikes for testing
    spike_samps = np.load(ks_dir.joinpath('spike_times.npy')).ravel()
    spike_clusters = np.load(ks_dir.joinpath('spike_clusters.npy')).ravel()
    params_fn = ks_dir.joinpath('params.py')
    with open(params_fn,'r') as fid:
        for line in fid:
            line_parts = line.split(' ')
            if line_parts[0] =='sample_rate':
                sample_rate = float(line_parts[-1])
    
    spike_times = spike_samps/sample_rate
    cluster_ids = np.unique(spike_clusters)
    cluster_ids.sort()

    # Load opto times and logs
    opto_df = pd.read_csv(opto_fn,sep='\t',index_col=0)
    log_df = pd.read_csv(log_fn,index_col=0,sep='\t')

    # Extract only tag times
    tags = extract_tagging_from_logs(log_df,opto_df)
    tag_times = tags['on_sec_corrected'].values
    tag_duration = tags['dur_sec'].mean()

    # Compute SALT data
    p_stat,I_stat = run_salt(spike_times,spike_clusters,cluster_ids,tag_times,
                             stim_duration=tag_duration)
    
    # Compute heuristic data
    n_stims_with_spikes,base_rate,stim_rate = compute_tagging_summary(spike_times,spike_clusters,cluster_ids,tag_times)

    # Export to a tsv
    salt_rez = pd.DataFrame()
    salt_rez['cluster_id']=cluster_ids
    salt_rez['salt_p_stat'] = p_stat
    salt_rez['salt_I_stat'] = I_stat
    salt_rez['n_stims_with_spikes'] = n_stims_with_spikes
    salt_rez['base_rate'] = base_rate
    salt_rez['stim_rate'] = stim_rate

    save_fn = ks_dir.joinpath('clusters.optotagging.tsv')
    salt_rez.to_csv(save_fn,sep='\t')
    print(f'optotagging info saved to {save_fn}.')

if __name__ == '__main__':
    main()