import sys
import tqdm
from pathlib import Path
import glob
import spikeinterface.toolkit as st
import numpy as np
import pandas as pd
import os
import click
import scipy.signal
sys.path.append('./src')
sys.path.append('./dynaresp')
sys.path.append('.')
sys.path.append('..')
import spikeinterface.sorters as ss
import spikeinterface.extractors as se
from readSGLX import *
from utils.ephys.resp_sig_proc import bwfilt,integrator

imec_dat = glob.glob(os.path.join(g_path, '*ap.bin'))[0]

def get_rect_int(imec_dat,chan_skip=10,downsample=10,integration_time=0.016):
    '''
    Performs downsampling by factor of 10 by default
    :param imec_dat:
    :return:
    '''
    meta = readMeta(Path(imec_dat))
    fs = SampRate(meta)
    rec = se.SpikeGLXRecordingExtractor(imec_dat)
    nchan = rec.get_num_channels()
    nsamps = rec.get_num_frames()
    t_vals = np.linspace(0,nsamps/fs,nsamps)[::downsample]
    chans = np.arange(0,nchan,chan_skip).astype('int')
    all_int = np.empty([len(chans),t_vals.shape[0]])

    for ii,chan in enumerate(tqdm.tqdm(chans)):
        temp = rec.get_traces(channel_ids=chan)
        temp_v = GainCorrectIM(temp,[chan],meta).ravel()
        temp_vf = bwfilt(temp_v,fs,300,10000)
        all_int[ii,:] = integrator(temp_vf,fs,span=integration_time)[::downsample]

    df = pd.DataFrame(all_int.T,columns=chans)
    df.index = t_vals
    df.index.name = 't'

    return(df)






