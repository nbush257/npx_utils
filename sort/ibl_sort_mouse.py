import click
import pykilosort
from pathlib import Path
import shutil
from viz import reports
import spikeglx
import pykilosort
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params
from pathlib import Path
from brainbox.metrics.single_units import spike_sorting_metrics
import one.alf.io as aio
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

DELETE=True

SCRATCH_DIR = Path('/active/ramirez_j/ramirezlab/nbush/scratch_ibl_sort')

def run_one(bin_file):
	INTEGRATION_DATA_PATH = bin_file.parent


	# # RUN PYKILOSORT
	alf_path = INTEGRATION_DATA_PATH.joinpath('alf')

	label = ""
	override_params = {}

	cluster_times_path = INTEGRATION_DATA_PATH.joinpath("cluster_times")
	ks_output_dir = INTEGRATION_DATA_PATH.joinpath(f"{pykilosort.__version__}" + label, bin_file.name.split('.')[0])
	ks_output_dir.mkdir(parents=True, exist_ok=True)

	params = ibl_pykilosort_params(bin_file)
	for k in override_params:
		params[k] = override_params[k]

	# run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=SCRATCH_DIR, params=params,ks_output_dir=ks_output_dir, alf_path=alf_path,log_level='DEBUG')

	if DELETE == False:
		working_directory = SCRATCH_DIR.joinpath('.kilosort', bin_file.stem)
		pre_proc_file = working_directory.joinpath('proc.dat')
		intermediate_directory = ks_output_dir.joinpath('intermediate')
		intermediate_directory.mkdir(exist_ok=True)
		shutil.copy(pre_proc_file, intermediate_directory)


	reports.qc_plots_metrics(bin_file=bin_file, pykilosort_path=alf_path, raster_plot=True, raw_plots=False, summary_stats=True,
	                         raster_start=100., raster_len=20., raw_start=50., raw_len=0.05,
	                         vmax=0.05, d_bin=5, t_bin=0.007)
	# Save metrics
	spikes = aio.load_object(alf_path, 'spikes')
	df_units, drift = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths)
	df_units.to_csv(alf_path.joinpath('metrics.csv'))


@click.command()
@click.argument('mouse_path')
def main(mouse_path):
	mouse_path = Path(mouse_path)
	mouse_id = mouse_path.name
	ap_binaries = list(mouse_path.rglob('*.ap.bin'))
	ap_binaries.sort()
	for bin_file in ap_binaries:
		try:
			run_one(bin_file)
		except:
			pass
	


if __name__=='__main__':
	main()
