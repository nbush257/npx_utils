import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
try:
    sys.path.append('Y:/helpers/NeuropixelsRegistration/python')
    sys.path.append('Y:/helpers/')
except:
    sys.path.append('/active/ramirez_j/ramirezlab/nbush/helpers/NeuropixelsRegistration/python')
    sys.path.append('/active/ramirez_j/ramirezlab/nbush/helpers/')
import estimate_displacement as ed
from numpy.fft import fft2, ifft2, fftshift, ifftshift # Python DFT
import pywt
from ibllib.io import spikeglx
from scipy.ndimage import gaussian_filter1d
from NeuropixelsRegistration.python import utils
from utils import mat2npy

# chan_map_fn = r"Y:\projects\dynaresp\data\processed\m2021-31\catgt_m2021-31_g0\m2021-31_g0_imec0\m2021-31_g0_tcat.imec0.ap_chanMap.mat"
chan_map_fn = r"Y:\projects\dynaresp\data\processed\m2021-10\catgt_m2021-10_g2\m2021-10_g2_imec0\m2021-10_g2_tcat.imec0.ap_chanMap.mat"
geomarray = mat2npy(chan_map_fn)
# bin_fn = r"Y:\projects\dynaresp\data\processed\m2021-31\catgt_m2021-31_g0\m2021-31_g0_imec0\m2021-31_g0_tcat.imec0.ap.bin"
bin_fn = r"Y:\projects\dynaresp\data\processed\m2021-10\catgt_m2021-10_g2\m2021-10_g2_imec0\m2021-10_g2_tcat.imec0.ap.bin"
reader = spikeglx.Reader(bin_fn)
total_shift = ed.estimate_displacement(reader, geomarray,
                                       reader_type='spikeglx')