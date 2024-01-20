import shutil
from pathlib import Path
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params, download_test_data

scratch_dir = Path('/active/ramirez_j/ramirezlab/nbush/scratch_ibl_sort')

data_path = Path("/archive/ramirez_j/ramirezlab/nbush/projects/test_ibl_sort")  # path on which the raw data will be downloaded
ks_output_dir = Path("/archive/ramirez_j/ramirezlab/nbush/projects/test_ibl_sort/out")  # path containing the kilosort output unprocessed
alf_path = ks_output_dir.joinpath('alf')  # this is the output standardized as per IBL standards (SI units, ALF convention)

# download the integration test data from amazon s3 bucket
bin_file, meta_file = download_test_data(data_path)

# prepare and mop up folder architecture for consecutive runs
DELETE = True  # delete the intermediate run products, if False they'll be copied over to the output directory for debugging

scratch_dir.mkdir(exist_ok=True)
ks_output_dir.mkdir(parents=True, exist_ok=True)

# loads parameters and run
params = ibl_pykilosort_params(bin_file)
params['Th'] = [6, 3]
run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=scratch_dir,
                      ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='INFO', params=params)

print('\tplotting QC')
reports.qc_plots_metrics(bin_file=bin_file, pykilosort_path=alf_path, raster_plot=False, raw_plots=False, summary_stats=False,
                            raster_start=5., raster_len=100., raw_start=5., raw_len=0.04,
                            vmax=0.5, d_bin=10, t_bin=0.007)
print('Saving qc metrics')
print('\n')
# Save metrics
spikes = aio.load_object(alf_path, 'spikes')
df_units, drift = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths)
df_units.to_csv(alf_path.joinpath('clusters.metrics.csv'))