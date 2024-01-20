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
def stash_problematic_files(phy_path):
    # NEED TO REMOVE OVERLAP SUMMARY
    overlap_file = phy_path.joinpath('overlap_summary.csv')
    events_file = phy_path.joinpath('events.csv')
    temp_dir = phy_path.joinpath('.temp')
    temp_dir.mkdir(exist_ok=True)
    if overlap_file.exists():        
        shutil.move(str(overlap_file),str(phy_path.joinpath('.temp')))
    if events_file.exists():        
        shutil.move(str(events_file),str(phy_path.joinpath('.temp')))

def unstash_problematic_files(phy_path):
    pass

phy_path = Path(r'D:\m2023-23_g0_imec0\imec0_ks2')
recording = phy_path.parent.glob('*.ap')


stash_problematic_files(phy_path)
sort_pyks = si.read_kilosort(phy_path)


we = si.extract_waveforms(recording,sort_pyks,sparse=False,folder=WVFM_PATH,overwrite=True,**job_kwargs)

    

# Compute metrics 
metrics = si.compute_quality_metrics(waveform_extractor=we,load_if_exists=True)
if USE_TEMPLATE_METRICS:
    template_metrics = si.compute_template_metrics(we,load_if_exists=True)

spike_locations = si.compute_spike_locations(we, method="center_of_mass", load_if_exists=True, **job_kwargs)

# Perform automated quality control
query = f'amplitude_cutoff<{AMPLITUDE_CUTOFF} & sliding_rp_violation<{SLIDING_RP} & amplitude_median>{AMP_THRESH}'
good_unit_ids = metrics.query(query).index
metrics['group'] = 'mua'
metrics.loc[good_unit_ids,'group']='good'
# metrics["peak_chan"] = pd.Series(si.get_template_extremum_channel(we, peak_sign="neg", outputs="index"))
if USE_TEMPLATE_METRICS:
    metrics = pd.concat([metrics,template_metrics],axis=1)

# Phy will not work if sparsity is incorrect (i.e., if computed waveforms are sparse, we need a sparsity object) (Does it not need a sort?)
si.export_to_phy(waveform_extractor=we,output_folder=PHY_DEST,use_relative_path=False,copy_binary=False,compute_pc_features=False,**job_kwargs)

print('Getting suggested merges')
auto_merge_candidates = si.get_potential_auto_merge(we)
pd.DataFrame(auto_merge_candidates).to_csv(PHY_DEST.joinpath('potential_merges.tsv'),sep='\t')

print('making driftmap')
# ---- Make driftmap ---- 
m = model.TemplateModel(dir_path=pyks_path,
                        dat_path=pyks_path.parent.joinpath('recording.dat'), 
                        sample_rate=sr,
                        n_channels_dat=recording.get_num_channels())