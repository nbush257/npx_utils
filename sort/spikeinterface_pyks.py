import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre 
import spikeinterface.sorters as ss 
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp 
import spikeinterface.sortingcomponents.motion_interpolation as sim
import spikeinterface.full as si
from pathlib import Path
from brainbox.metrics import single_units
import numpy as np
import pandas as pd
from ibllib.ephys import spikes
from brainbox.plot import driftmap
import matplotlib.pyplot as plt
import shutil
from phylib.io import model
import click
import re
import logging
import sys
import os
import time

# May want to do a "remove duplicate spikes" after manual sorting  - this would allow manual sorting to merge  units that have some, but not a majority, of duplicated spikes

#TODO: Maybe sparsity fails with too few spikes?
#TODO: I/O handling 
#TODO: Compress raw recording for storage
#TODO: Make a CLI
#TODO: Manage folder structure
#TODO: Make work with multiple probes
#TODO: Test on only one recording
#TODO: make sequence file
#TODO: Make scratch directory
#TODO: Transfer all data to final destination on ARCHIVE 
#TODO: Document
#TODO: Refactor into smaller functions?
#TODO: Log sort progress?

job_kwargs = dict(chunk_duration="1s", n_jobs=15, progress_bar=True)
we_kwargs = dict()

# QC presets
AMPLITUDE_CUTOFF = 0.1
SLIDING_RP = 0.1
AMP_THRESH = 40.
MIN_SPIKES = 500

RUN_PC = False
MOTION_PRESET ='kilosort_like' # 'kilosort_like','nonrigid_accurate'
SORTER = 'kilosort3' # 'pykilosort','kilosort3'

if sys.platform == 'win32':
    SCRATCH_DIR = Path(r'D:/si_temp2')
    KS3_ROOT = r'C:\Users\nbush\helpers\kilosort3'
    IS_HPC=False
else:
    import joblib
    SCRATCH_DIR = Path('/gpfs/home/nbush/si_scratch')
    job_kwargs = dict(chunk_duration="1s", n_jobs=joblib.cpu_count(), progress_bar=True)
    import matplotlib
    matplotlib.use('Agg')
    IS_HPC=True

def print_elapsed_time(start_time):
    print(f'Elapsed time: {time.time()-start_time:0.0f} seconds')


def run_probe(probe_src,stream,probe_local,testing=False):
    start_time = time.time()
    if not IS_HPC:
        si.Kilosort3Sorter.set_kilosort3_path(KS3_ROOT)

    # Set paths
    PHY_DEST = probe_local.joinpath('phy_output')
    SORT_PATH = probe_local.joinpath(f'.ks3')
    SORT_PATH = probe_local.joinpath(f'.pyks')
    WVFM_PATH = SORT_PATH.joinpath('.wvfm')
    PREPROC_PATH = probe_local.joinpath('.preproc')
    MOTION_PATH = probe_local.joinpath('.motion')
    probe_local.mkdir(parents=True,exist_ok=True)
    if PHY_DEST.exists():
        print(f'Local phy destination exists ({PHY_DEST}). Skipping this probe {probe_src}')
        return

    # SET SORT PARAMETERS
    sorter_params = {}
    if SORTER == "kilosort3":
        sorter_params['car'] = False
        sorter_params['do_correction'] = False


    # POINT TO RECORDING and concatenate
    if not PREPROC_PATH.exists():
        recording = se.read_spikeglx(probe_src,stream_id = stream)
        print(recording)
        rec_list =[]
        curr_samples = 0
        sample_bounds = []
        for ii in range(recording.get_num_segments()):
            seg = recording.select_segments(ii)
            sr = seg.get_sampling_frequency()
            # THIS LINE CUTS THE SEGMENTS
            if testing:
                use_secs = 60
                print(f"TESTING: ONLY RUNNING ON {use_secs}s per segment")
                seg = seg.frame_slice(0,30000*use_secs)
            rec_list.append(seg)
            sample_bounds.append([curr_samples,curr_samples + seg.get_num_samples(),seg.sample_index_to_time(curr_samples),
                                  seg.sample_index_to_time(curr_samples+seg.get_num_samples())])
            curr_samples +=seg.get_num_samples()
        recording = si.concatenate_recordings(rec_list)
        sample_bounds = pd.DataFrame(sample_bounds,columns=['rec_start_sample','rec_end_sample','rec_start_time','rec_end_time'])
        sample_bounds.to_csv(probe_local.joinpath('sequence.tsv'),sep='\t')

        # Preprocessing
        print('Preprocessing IBL destripe...',end='')
        rec_filtered = spre.highpass_filter(recording)
        rec_shifted = spre.phase_shift(rec_filtered)
        bad_channel_ids, all_channels = spre.detect_bad_channels(rec_shifted)
        rec_interpolated = spre.interpolate_bad_channels(rec_shifted, bad_channel_ids)
        rec_destriped = spre.highpass_spatial_filter(rec_interpolated)    
        rec_mc = rec_destriped 
        print('done')
        if MOTION_PATH.exists():
            print('Motion info loaded')
            motion_info = si.load_motion_info(MOTION_PATH)
            rec_mc = sim.interpolate_motion(recording = rec_destriped,
                                                        motion=motion_info['motion'],
                                                        temporal_bins=motion_info['temporal_bins'],
                                                        spatial_bins=motion_info['spatial_bins'],
                                                        **motion_info['parameters']['interpolate_motion_kwargs'],
                                                        )
        else:
            print('Motion correction KS-like...',end='')
            rec_mc, motion_info = si.correct_motion(rec_destriped, preset=MOTION_PRESET,
                            folder=MOTION_PATH,
                            output_motion_info=True, **job_kwargs)
            print('done')
            print_elapsed_time(start_time)
        
        rec_mc.save(folder=PREPROC_PATH,**job_kwargs)     
        print_elapsed_time(start_time) 
    else:
        pass
    
    # Load preprocessed data on disk
    recording = si.load_extractor(PREPROC_PATH)
    print('Loading preprocessed data...',end='')
    sr = recording.get_sampling_frequency()
    print(f"loaded recording from {PREPROC_PATH}")
    
    # Plot motion data
    print('Loading motion info',end='')
    motion_info = si.load_motion_info(MOTION_PATH)
    if not MOTION_PATH.joinpath('driftmap.png').exists():
        fig = plt.figure(figsize=(14, 8))
        si.plot_motion(motion_info, figure=fig, 
                    color_amplitude=True, amplitude_cmap='inferno', scatter_decimate=10)
        plt.savefig(MOTION_PATH.joinpath('driftmap.png'),dpi=300)
        for ax in fig.axes[:-1]:
            ax.set_xlim(30,60)
        plt.savefig(MOTION_PATH.joinpath('driftmap_zoom.png'),dpi=300)
    print('loaded')
    print_elapsed_time(start_time) 


    # Run sorter
    if SORT_PATH.exists():
        print('='*100)
        print('Found sorting. Loading...',end='')
        sort_rez = si.read_kilosort(SORT_PATH.joinpath('sorter_output/output'))
        print('loaded.')
    else:
        if IS_HPC:
            sort_rez = ss.run_sorter(SORTER,recording, output_folder=SORT_PATH,verbose=True,singularity_image=True,**sorter_params,**job_kwargs)
        else:
            sort_rez = ss.run_sorter(SORTER,recording, output_folder=SORT_PATH,verbose=True,**sorter_params,**job_kwargs)
    print_elapsed_time(start_time) 

    temp_wh_fn = SORT_PATH.joinpath('sorter_output/output/temp_wh.dat')
    if temp_wh_fn.exists():
        print(f'Removing KS temp_wh.dat: {temp_wh_fn}')
        os.remove(temp_wh_fn)
    
    if WVFM_PATH.exists():
        print('Found waveforms. Loading',end='...')
        we = si.load_waveforms(WVFM_PATH)
        sort_rez = we.sorting
        print('loaded')

    else:
        print('Extracting waveforms...',end='')
        we = si.extract_waveforms(recording,sort_rez,folder=WVFM_PATH,sparse=False,**job_kwargs)
        print('done')
        print('Removing redundant spikes...',end='')
        sort_rez = si.remove_duplicated_spikes(sort_rez,censored_period_ms=0.166)
        sort_rez = si.remove_redundant_units(sort_rez,align=False,remove_strategy='max_spikes')
        we = si.extract_waveforms(recording,sort_rez,sparse=False,folder=WVFM_PATH,overwrite=True,**job_kwargs)
        print('done')
    print('Comuting sparsity',end='')
    sparsity = si.compute_sparsity(we,num_channels=9)
    print('done')
    print_elapsed_time(start_time) 

    # COMPUTE METRICS
    # Needs fewer jobs to avoid a memory error?
    amplitudes = si.compute_spike_amplitudes(we,load_if_exists=True, n_jobs=4,chunk_duration='500ms')
    if RUN_PC:
        pca = si.compute_principal_components(waveform_extractor=we, n_components=5, load_if_exists=True,mode='by_channel_local',**job_kwargs,sparsity=sparsity)
    

    # Compute metrics 
    print('Computing metrics')
    metrics = si.compute_quality_metrics(waveform_extractor=we,load_if_exists=True)
    print('\ttemplate metrics')
    template_metrics = si.compute_template_metrics(we,load_if_exists=True)

    # # Perform automated quality control
    query = f'amplitude_cutoff<{AMPLITUDE_CUTOFF} & sliding_rp_violation<{SLIDING_RP} & amplitude_median>{AMP_THRESH}'
    good_unit_ids = metrics.query(query).index
    metrics['group'] = 'mua'
    metrics.loc[good_unit_ids,'group']='good'

    print_elapsed_time(start_time) 
    print("Exporting to phy")
    # Needs fewer jobs to avoid a memory error?
    si.export_to_phy(waveform_extractor=we,output_folder=PHY_DEST,
                    # sparsity=sparsity,
                    use_relative_path=True,copy_binary=True,
                    compute_pc_features=False,**job_kwargs)

    print('Getting suggested merges')
    auto_merge_candidates = si.get_potential_auto_merge(we)
    pd.DataFrame(auto_merge_candidates).to_csv(PHY_DEST.joinpath('potential_merges.tsv'),sep='\t')

    # TODO: check for optotagging, bombcell?
    print('Done sorting!')
    print_elapsed_time(start_time) 


@click.command()
@click.argument('mouse_path')
@click.option('--dest','-d',default=None)
@click.option('--testing',is_flag=True)
@click.option('--move_final',is_flag=True)
def main(mouse_path,dest,testing,move_final):
    cleanup_local = True
    mouse_path = Path(mouse_path)
    run_name = mouse_path.name
    gate_list = list(mouse_path.glob(f'*{run_name}*'))
    run_local = SCRATCH_DIR.joinpath(run_name+'_si-sort')
    if move_final:
        print(f'WARNING: will remove {run_local} after sorting')
    print(f'Gate List: {gate_list}')
    
    for gate_path in gate_list:
        if dest is not None:
            run_dest = Path(dest)
        else:
            run_dest = gate_path.parent

        gate_local = run_local.joinpath(gate_path.name).joinpath('si-sort')
        gate_dest = run_dest.joinpath(gate_path.name)
        
        probe_list = list(gate_path.glob(f'*{run_name}*imec*'))
        
        for probe_path in probe_list:
            stream = re.search('imec[0-9]',str(probe_path)).group()+'.ap'
            probe_local = gate_local.joinpath(''+stream[:-3])
            probe_dest = gate_dest.joinpath('si-sort')

            print('='*100)
            print('Running SI kilosort:')
            print(f"\tGate: {gate_path}")
            print(f"\tStream: {stream}")
            print('='*100)
            run_probe(gate_path,stream,probe_local,testing=testing)

            if move_final:
                if probe_dest.exists():
                    print(f'WARNING: Not moving because target {probe_dest} already exists')
                    cleanup_local = False
                else:
                    print(f'Moving sorted data from {probe_local} to {probe_dest}')
                    probe_dest.mkdir(parents=True,exist_ok=False)
                    shutil.move(str(probe_local),str(probe_dest))
    if move_final & cleanup_local:
        print(f"Removing local {run_local}")
        shutil.rmtree(str(run_local))



if __name__=='__main__':
    main()