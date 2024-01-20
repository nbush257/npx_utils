from pathlib import Path
import spikeinterface.full as si
AMPLITUDE_CUTOFF = 0.1
SLIDING_RP = 0.1
AMP_THRESH = 50.
RUN_PC = False
USE_SPARSE = True
USE_TEMPLATE_METRICS = False
PLOT_DRIFTMAP =False

PREPROC_PATH  = Path(r'D:\si_temp\m2023-24_si-sort\m2023-24_g0\si-sort\imec0\.preproc')
KS3_path = Path(r'D:\si_temp\m2023-24_si-sort\m2023-24_g0\si-sort\imec0\.ks3')
WVFM_PATH = KS3_path.joinpath('.wvfm')
PHY_DEST = KS3_path.joinpath("phy_output")

sorter_params = {}
sorter_params['car'] = False
sorter_params['do_correction']=False
job_kwargs = dict(chunk_duration="500ms", n_jobs=1, progress_bar=True)
si.Kilosort3Sorter.set_kilosort3_path(r'C:\Users\nbush\helpers\kilosort3')

recording = si.load_extractor(PREPROC_PATH)
# sort_ks3 = si.run_sorter('kilosort3',recording,KS3_path,verbose=True,**sorter_params,**job_kwargs)
print("Load sort")
sort_ks3 = si.read_kilosort(KS3_path.joinpath('sorter_output'))
print("remove redundant")

# sort_ks3 = si.remove_duplicated_spikes(sort_ks3,censored_period_ms=0.166)
# sort_ks3 = si.remove_redundant_units(sort_ks3,align=False,remove_strategy='max_spikes')
print("Extract wvfm")
we = si.extract_waveforms(recording,sort_ks3,folder=WVFM_PATH,**job_kwargs)
we = si.load_waveforms(WVFM_PATH)


print('Metrics')
pyks_path = Path(sort_ks3.get_annotation('phy_folder'))
amplitudes = si.compute_spike_amplitudes(we,load_if_exists=True, n_jobs=6,chunk_duration='500ms')
metrics = si.compute_quality_metrics(waveform_extractor=we,load_if_exists=True)
spike_locations = si.compute_spike_locations(we, method="center_of_mass", load_if_exists=True, **job_kwargs)

# Perform automated quality control
query = f'amplitude_cutoff<{AMPLITUDE_CUTOFF} & sliding_rp_violation<{SLIDING_RP} & amplitude_median>{AMP_THRESH}'
good_unit_ids = metrics.query(query).index
metrics['group'] = 'mua'
metrics.loc[good_unit_ids,'group']='good'


print("Exporting to phy")
# Phy will not work if sparsity is incorrect (i.e., if computed waveforms are sparse, we need a sparsity object) (Does it not need a sort?)
# Needs fewer jobs to avoid a memory error?
si.export_to_phy(waveform_extractor=we,output_folder=PHY_DEST,use_relative_path=False,copy_binary=False,compute_pc_features=False,n_jobs=4,chunk_duration='500ms')
