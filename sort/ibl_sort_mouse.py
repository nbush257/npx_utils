#TODO: Clean up
#TODO: Make CLI work OR gui file chooser?
#TODO: Remove test paths
#TODO: Test postprocessing
#TODO: Test QC/metrics - maybe use spikeinterface?
#TODO: Test with multiple probes, multiple trials
#TODO: Test with multiple probes, single trials
#TODO: Test without compression
# TODO: Move files from local to remote
#TODO: Put compression afterward (Optionally?) - slows down the processing
import click
from pathlib import Path
import shutil
from viz import reports
import spikeglx
import pykilosort
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params, _get_multi_parts_records
from pathlib import Path
from brainbox.metrics.single_units import spike_sorting_metrics
import one.alf.io as aio
import shutil
import matplotlib.pyplot as plt
import matplotlib
import sys
from iblutil.io import hashfile, params
from pykilosort import add_default_handler, run, Bunch, __version__
from pykilosort.params import KilosortParams
import logging
import datetime
from ibllib.ephys import spikes
_logger = logging.getLogger("pykilosort")
log_level='INFO'

if sys.platform=='win32':
	SCRATCH_DIR = Path(r'D:/pyks_datatemp')
	TARGET_DIR = Path(r'Z:\test_ibl_sort')

else:
    SCRATCH_DIR = Path('/gpfs/home/nbush/si_scratch')
    TARGET_DIR = Path('/active/ramirez_j/ramirezlab/nbush/projects/iso_npx/test')
    import matplotlib
    matplotlib.use('Agg')
DELETE=True




def run_one(bin_file,TARGET_DIR,log_file):
	label = ""
	override_params={}

	alf_dir = TARGET_DIR.joinpath('alf')
	ks_output_dir = TARGET_DIR.joinpath(f"{pykilosort.__version__}" + label)
	ks_output_dir.mkdir(parents=True, exist_ok=True)

	
	if isinstance(bin_file,list):
		params = ibl_pykilosort_params(bin_file[0]) 
		data_dir = bin_file[0].parent
	else:
		params = ibl_pykilosort_params(bin_file) 
		data_dir = bin_file.parent
	for k in override_params:
		params[k] = override_params[k]
	
	# Modified preprocess lin 424 to limit the number of processes to 8
	try:
		_logger.info(f"Starting Pykilosort version {__version__}")
		_logger.info(f"Scratch dir {SCRATCH_DIR}")
		_logger.info(f"Output dir {ks_output_dir}")
		_logger.info(f"Data dir {data_dir}")
		_logger.info(f"Log file {log_file}")
		_logger.info(f"Loaded probe geometry for NP{params['probe']['neuropixel_version']}")

		run(bin_file, dir_path=SCRATCH_DIR, output_dir=ks_output_dir, **params)

		for qc_file in SCRATCH_DIR.rglob('_iblqc_*'):
			shutil.copy(qc_file, ks_output_dir.joinpath(qc_file.name))
		if DELETE:
			shutil.rmtree(SCRATCH_DIR.joinpath(".kilosort"), ignore_errors=True)
	except Exception as e:
		_logger.exception("Error in the main loop")
		raise e

	# move the log file and all qcs to the output folder
	shutil.move(log_file, ks_output_dir.joinpath('spike_sorting_pykilosort.log'))

	s2v = _sample2v(bin_file)
	alf_dir.mkdir(exist_ok=True, parents=True)

	#TODO: multiple trials (concatenated) do not translate to alf such that the original file is kept, and the phy params file is no good.
	spikes.ks2_to_alf(ks_output_dir, bin_file[0].parent, alf_dir, ampfactor=s2v)
	if DELETE == False:
		_logger.info("You are likely to run into path errors here. NEB assumes you wont be saving the intermediate steps")
		working_directory = SCRATCH_DIR.joinpath('.kilosort', bin_file.stem)
		pre_proc_file = working_directory.joinpath('proc.dat')
		intermediate_directory = ks_output_dir.joinpath('intermediate')
		intermediate_directory.mkdir(exist_ok=True)
		shutil.copy(pre_proc_file, intermediate_directory)
	print('\tplotting QC')
	reports.qc_plots_metrics(bin_file=bin_file, pykilosort_path=alf_path, raster_plot=True, raw_plots=True, summary_stats=True,
	                         raster_start=100., raster_len=300., raw_start=100., raw_len=0.04,
	                         vmax=0.5, d_bin=10, t_bin=0.1)
	print('Saving qc metrics')
	print('\n')
	# Save metrics
	spikes = aio.load_object(alf_path, 'spikes')
	df_units, drift = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths)
	df_units.to_csv(alf_path.joinpath('clusters.metrics.tsv'))
	[_logger.removeHandler(h) for h in _logger.handlers]
	print("Done")


def run_compression(bin_files,keep_original=True):
	'''
	Run MTSCOMP on the bin files. Checks to see if the cbin files have already been computed.
	Returns a list of compressed files.
	'''
	cbin_files = []
	for bin_file in bin_files:
		f_out = bin_file.with_suffix('.cbin')
		if f_out.exists():
			sr = spikeglx.Reader(f_out)
			if sr.is_mtscomp:
				_logger.info(f'File {bin_file} is already compressed as {f_out}. Skipping compression.')
				cbin_files.append(f_out)
			else:
				_logger.error(f'File {f_out} already exists but is not a mtscomp bin file.')
		else:
			sr = spikeglx.Reader(bin_file)
			_logger.info(f'Compressing file {bin_file}.')
			f_out = sr.compress_file(keep_original=keep_original)
			cbin_files.append(f_out)
	return(cbin_files)

def _sample2v(ap_file):
	if isinstance(ap_file,list):
		ap_file = ap_file[0]
	md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
	s2v = spikeglx._conversion_sample2v_from_meta(md)
	return s2v["ap"][0]

@click.command()
@click.argument('gate_path')
@click.option('--skip_compression','-S',is_flag=True,help='Set this flag if you want to skip compression using MTSCOMP (Not recommended)')
def main(gate_path,skip_compression):
	START_TIME = datetime.datetime.now()
	log_file = SCRATCH_DIR.joinpath(f"_{START_TIME.isoformat().replace(':','-')}_kilosort.log")
	add_default_handler(level=log_level)
	add_default_handler(level=log_level, filename=log_file)

	gate_path = Path(gate_path)
	prb_folders = list(gate_path.glob('*imec*'))
	assert len(prb_folders)>0,f'No recordings found in {gate_path}'
	for prb in prb_folders:
		run_data = None
		bin_files = list(prb.glob('*ap.bin'))
		bin_files.sort()

		if len(bin_files)==0:
			cbin_files = list(prb.glob('*ap.cbin'))
			if len(cbin_files)==0:
				_logger.error('No bin files or cbin files found. Exiting')
			else:
				_logger.info(f'No bin files found, found {len(cbin_files)} cbin files. Using these')
				run_data = cbin_files
		
		# Compute compression if requested
		if not skip_compression and run_data is None:
			cbin_files = run_compression(bin_files,keep_original=True)
			run_data = cbin_files
		elif run_data is None:
			_logger.info('Running without compression...')
			run_data = bin_files
		else:
			pass
		
		# RUN
		_logger.info(f'Running on {run_data}')
		run_one(run_data,TARGET_DIR,log_file)

		
		
	


if __name__=='__main__':
	main()
