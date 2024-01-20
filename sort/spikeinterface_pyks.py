import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre 
import spikeinterface.sorters as ss 
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp 
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

# May want to do a "remove duplicate spikes" after manual sorting  - this would allow manual sorting to merge  units that have some, but not a majority, of duplicated spikes

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

job_kwargs = dict(chunk_duration="500ms", n_jobs=10, progress_bar=True)

# QC presets
AMPLITUDE_CUTOFF = 0.1
SLIDING_RP = 0.1
AMP_THRESH = 50.
RUN_PC = False
USE_SPARSE = True
USE_TEMPLATE_METRICS = False
PLOT_DRIFTMAP =False

if sys.platform == 'win32':
    SCRATCH_DIR = Path(r'D:/si_temp')
else:
    import joblib
    SCRATCH_DIR = Path('/gpfs/home/nbush/si_scratch')
    job_kwargs = dict(chunk_duration="1s", n_jobs=joblib.cpu_count(), progress_bar=True)
    import matplotlib
    matplotlib.use('Agg')


def run_probe(probe_src,stream,probe_local,testing=False):
    # Set paths
    PHY_DEST = probe_local.joinpath('phy_output')
    SORT_PATH = probe_local.joinpath(f'.pyks25')
    WVFM_PATH = probe_local.joinpath('.wvfm')
    PREPROC_PATH = probe_local.joinpath('.preproc')
    DEDUPE_PATH = probe_local.joinpath('.pyks25_dedupe')
    MOTION_PATH = probe_local.joinpath('.motion')

    # SET SORT PARAMETERS
    sorter_params ={}
    # sorter_params['perform_drift_registration'] = False

    # POINT TO RECORDING and concatenate
    if not PREPROC_PATH.exists():
        recording = se.read_spikeglx(probe_src,stream_id = stream)
        print(recording)
        rec_list =[]
        for ii in range(recording.get_num_segments()):
            seg = recording.select_segments(ii)
            sr = seg.get_sampling_frequency()
            # THIS LINE CUTS THE SEGMENTS
            if testing:
                use_secs = 60
                print(f"TESTING: ONLY RUNNING ON {use_secs}s per segment")
                seg = seg.frame_slice(0,30000*use_secs)
            rec_list.append(seg)
        recording = si.concatenate_recordings(rec_list)


        # Set up preprocessing
        recording = spre.phase_shift(recording)
        recording = spre.highpass_filter(recording, freq_min=300)
        recording = spre.common_reference(recording,reference='global',operator='median')
        recording = spre.highpass_spatial_filter(recording,n_channel_pad=60)

        # Drift correction explicit
        recording, motion_info = si.correct_motion(recording, preset='kilosort_like',
                                folder=MOTION_PATH,
                                output_motion_info=True, **job_kwargs)
        recording.save(folder=PREPROC_PATH,**job_kwargs)

    else:
        print('Loading preprocessed data...')

    recording = si.load_extractor(PREPROC_PATH)
    sr = recording.get_sampling_frequency()
    print(f"loaded recording from {PREPROC_PATH}")
    
    # Motion data
    print('Loading motion info')
    motion_info = si.load_motion_info(MOTION_PATH)
    if  not MOTION_PATH.joinpath('driftmap.png').exists():
        fig = plt.figure(figsize=(14, 8))
        si.plot_motion(motion_info, figure=fig, 
                    color_amplitude=True, amplitude_cmap='inferno', scatter_decimate=10)
        plt.savefig(MOTION_PATH.joinpath('driftmap.png'),dpi=300)
        for ax in fig.axes[:-1]:
            ax.set_xlim(30,60)
        plt.savefig(MOTION_PATH.joinpath('driftmap_zoom.png'),dpi=300)
    print('loaded')
    

    # Run sorter

    if SORT_PATH.exists():
        print('='*100)
        print('Found sorting. Loading...')
        sort_pyks = si.read_kilosort(SORT_PATH.joinpath('sorter_output/output'))
        print('loaded.')
    else:
        sort_pyks = ss.run_sorter('pykilosort',recording, output_folder=SORT_PATH,verbose=True,**sorter_params,**job_kwargs)
        print('Deleting temporary sort files...')
        files_to_move = SORT_PATH.rglob('_iblqc_*')
        for fn in files_to_move:
            shutil.copy(fn,SORT_PATH.joinpath(fn.name))
        # --- CLEAN UP FROM SORTING --- #

        # delete kilosort temp paths
        temp_path = list(SORT_PATH.rglob('.kilosort'))[0]
        shutil.rmtree(temp_path)
        print('deleted.')
    
    print('Removing redundant spikes...')

    # sort_pyks = si.remove_duplicated_spikes(sort_pyks,censored_period_ms=0.166)
    # sort_pyks = si.remove_redundant_units(sort_pyks,align=False,remove_strategy='max_spikes')

    

    print('='*100)
    if WVFM_PATH.exists():
        print('='*100)
        print('Waveforms found. Loading...')
        we = si.load_waveforms(WVFM_PATH)
    else:
        print('Extracting waveforms')
        we = si.extract_waveforms(recording,sort_pyks,folder=WVFM_PATH,method='best_channels',num_channels=8,**job_kwargs)
    
    
    pyks_path = Path(sort_pyks.get_annotation('phy_folder'))

    
    # COMPUTE METRICS
    print('Computing metrics')
    # Needs fewer jobs to avoid a memory error?
    amplitudes = si.compute_spike_amplitudes(we,load_if_exists=True, n_jobs=6,chunk_duration='500ms')
    if RUN_PC:
        pca = si.compute_principal_components(waveform_extractor=we, n_components=5, load_if_exists=True,mode='by_channel_local')
    


    # Compute metrics 
    # metrics = si.compute_quality_metrics(waveform_extractor=we,load_if_exists=True)
    if USE_TEMPLATE_METRICS:
        template_metrics = si.compute_template_metrics(we,load_if_exists=True)

    # spike_locations = si.compute_spike_locations(we, method="center_of_mass", load_if_exists=True, **job_kwargs)

    # Perform automated quality control
    # query = f'amplitude_cutoff<{AMPLITUDE_CUTOFF} & sliding_rp_violation<{SLIDING_RP} & amplitude_median>{AMP_THRESH}'
    # good_unit_ids = metrics.query(query).index
    # metrics['group'] = 'mua'
    # metrics.loc[good_unit_ids,'group']='good'
    # metrics["peak_chan"] = pd.Series(si.get_template_extremum_channel(we, peak_sign="neg", outputs="index"))
    if USE_TEMPLATE_METRICS:
        metrics = pd.concat([metrics,template_metrics],axis=1)

    print("Exporting to phy")
    # Phy will not work if sparsity is incorrect (i.e., if computed waveforms are sparse, we need a sparsity object) (Does it not need a sort?)
    # Needs fewer jobs to avoid a memory error?
    si.export_to_phy(waveform_extractor=we,output_folder=PHY_DEST,use_relative_path=False,copy_binary=False,compute_pc_features=False,n_jobs=4,chunk_duration='500ms')
    
    print('Getting suggested merges')
    # auto_merge_candidates = si.get_potential_auto_merge(we)
    # pd.DataFrame(auto_merge_candidates).to_csv(PHY_DEST.joinpath('potential_merges.tsv'),sep='\t')

    print('making driftmap')
    # ---- Make driftmap ---- 
    if PLOT_DRIFTMAP:
        m = model.TemplateModel(dir_path=pyks_path,
                                dat_path=pyks_path.parent.joinpath('recording.dat'), 
                                sample_rate=sr,
                                n_channels_dat=recording.get_num_channels())

        depths_unc = m.get_depths()
        ts_unc = m.spike_times
        ts_corr = sort_pyks.to_spike_vector()['sample_index']/sort_pyks.get_sampling_frequency()
        drift = np.load(pyks_path.joinpath('drift.um.npy'))
        drift_times = np.load(pyks_path.joinpath('drift.times.npy'))

    
        f = plt.figure(figsize=(18,7))
        gs = f.add_gridspec(nrows=7,ncols=2)
        ax_drift = f.add_subplot(gs[0,0])
        ax_rast = f.add_subplot(gs[1:,0],sharex=ax_drift)
        ax_drift.plot(drift_times,drift)
        ax_drift.set_ylabel('Drift ($\mu$m)')
        driftmap(ts_unc,depths_unc,t_bin=0.01,ax=ax_rast)
        ax_drift.set_title('Uncorrected')

        ax_drift2 = f.add_subplot(gs[0,1],sharex=ax_drift,sharey=ax_drift)
        ax_rast2 = f.add_subplot(gs[1:,1],sharex=ax_drift,sharey=ax_rast)
        ax_drift2.plot(drift_times,drift)
        ax_drift2.set_ylabel('Drift ($\mu$m)')
        ax_drift2.set_title('Drift Corrected')
        driftmap(ts_corr,spike_locations['y'],t_bin=0.01,ax=ax_rast2)
        plt.tight_layout()
        plt.savefig(PHY_DEST.joinpath('driftmaps.png'),dpi=300,bbox_inches='tight')
        plt.close('all')


    # TODO: check for optotagging, bombcell?
    print('Done sorting!')

@click.command()
@click.argument('mouse_path')
@click.option('--dest','-d',default=None)
@click.option('--testing',is_flag=True)
@click.option('--move_final',is_flag=True)
def main(mouse_path,dest,testing,move_final):
    mouse_path = Path(mouse_path)
    run_name = mouse_path.name
    gate_list = list(mouse_path.glob(f'*{run_name}*'))
    run_local = SCRATCH_DIR.joinpath(run_name+'_si-sort')
    
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
            probe_dest = gate_dest.joinpath('si_'+probe_local.name)

            print('='*100)
            print('Running SI kilosort:')
            print(f"\tGate: {gate_path}")
            print(f"\tStream: {stream}")
            print('='*100)
            run_probe(gate_path,stream,probe_local,testing=testing)

            if move_final:
                if probe_dest.exists():
                    print(f'WARNING: Not moving because target {probe_dest} already exists')
                else:
                    print(f'Moving sorted data from {probe_local} to {probe_dest}')
                    probe_dest.mkdir(parents=True,exist_ok=False)
                    shutil.move(str(probe_local),str(probe_dest))


if __name__=='__main__':
    main()