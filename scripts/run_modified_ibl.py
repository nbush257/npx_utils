import logging
import os
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import parmap
import spikeglx

from detect.detector import Detect
from localization_pipeline.denoiser import Denoise

from detect.deduplication import deduplicate_gpu, deduplicate

from scipy.signal import argrelmin

from detect.run import run
import os
import numpy as np
from tqdm import tqdm
# from residual import RESIDUAL
from localization_pipeline.localizer import LOCALIZER
from localization_pipeline.merge_results import get_merged_arrays
## =============================================

geom_path ="/active/ramirez_j/ramirezlab/nbush/helpers/spike_localization_registration_hpc/channels_maps/np1_channel_map.npy"
path_nn_detector = "/active/ramirez_j/ramirezlab/nbush/helpers/spike_localization_registration_hpc/pretrained_detector/detect_np1.pt"
path_nn_denoiser = "/active/ramirez_j/ramirezlab/nbush/helpers/spike_localization_registration_hpc/pretrained_denoiser/denoise.pt"
standardized_path = '/archive/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/ibl_pipeline_test/m2021-32_g0_t0.imec0.ap.standardized.bin'
standardized_meta = standardized_path.replace('.bin','.meta')
standardized_dtype = 'float32'
standardized_path = Path(standardized_path)
meta = spikeglx.read_meta_data(Path(standardized_meta))
sampling_rate = float(meta['imSampRate'])
len_recording = int(meta['fileTimeSecs'])
len_recording = 1
detection_directory = standardized_path.parent.joinpath('detection_results_threshold')
print(standardized_path.stem)
geom_array = np.load(geom_path)
apply_nn = True ### If set to false, run voltage threshold instead of NN detector
spatial_radius = 70
n_sec_chunk = 1
n_processors = 1
n_sec_chunk_gpu_detect = .1
detect_threshold = 0.5 ## 0.5 if apply NN, 4/5/6 otherwise
n_filters_detect = [16, 8, 8]
spike_size_nn = 121 ### In sample steps
n_filters_denoise = [16, 8, 4]
filter_sizes_denoise = [5, 11, 21]

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

n_batches = len_recording//n_sec_chunk

print('Running detect')
run(standardized_path, standardized_dtype, detection_directory, geom_array, spatial_radius, apply_nn, n_sec_chunk, n_batches, n_processors, n_sec_chunk_gpu_detect, sampling_rate, len_recording,
    detect_threshold, path_nn_detector, n_filters_detect, spike_size_nn, path_nn_denoiser, n_filters_denoise, filter_sizes_denoise, run_chunk_sec='full')

#
# ===================================
print('Running localize')
bin_file = standardized_path
residual_file = bin_file

fname_spike_train = detection_directory.joinpath('spike_index.npy')
# Sort spike train if not
spt_array = np.load(fname_spike_train)
spt_array = spt_array[spt_array[:, 0].argsort()]
np.save(fname_spike_train, spt_array)

n_channels = geom_array.shape[0]

denoiser_weights = path_nn_denoiser
denoiser_min = 42 ## Goes with the weights


fname_templates = None

localizer_obj = LOCALIZER(bin_file, standardized_dtype, fname_spike_train, fname_templates, geom_path, denoiser_weights, denoiser_min,n_processors=n_processors)
# localizer_obj.get_offsets()
# localizer_obj.compute_aligned_templates()
localizer_obj.load_denoiser()
localize_dir = standardized_path.parent.joinpath('position_results')
if not os.path.exists(localize_dir):
    os.makedirs(localize_dir)
for i in tqdm(range(n_batches)):
    localizer_obj.get_estimate(i, threshold = detect_threshold, output_directory =localize_dir)