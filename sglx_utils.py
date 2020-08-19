import sys
import tqdm
import pandas as pd
sys.path.append('../')
import spikeinterface.extractors as se
from readSGLX import *
from utils.ephys.signal import *

def get_rect_int(imec_dat,chan_skip=10,downsample=10,integration_time=0.016):
    '''
    Get rectified and integrated trace across the imec probe
    Attempts to psuedo replace the need for an extracellular probe

    Parameters
    ----------
    imec_dat : filename of the imec AP data
    chan_skip : number of channels to skip (speeds up computation and reduces memory requirements)
    downsample : Downsample factor (speeds up computation and reduces memory requirements)
    integration_time : Window of exponential integration in seconds

    Returns
    -------

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
        temp_vf = butter_bandpass_filter(temp_v,300,10000,fs)
        all_int[ii,:] = integrator(temp_vf,fs,span=integration_time)[::downsample]

    df = pd.DataFrame(all_int.T,columns=chans)
    df.index = t_vals
    df.index.name = 't'

    return(df)






