'''
Run this script to extract onsets,offsets, durations, and amplitudes (in V) of the analog opto trace
returns a csv dataframe

# This is pretty hacky in the organization with the rest of the modules. NEB should refactor a lot of this code to work nicer with other software
# Runs in the iblenv environment
'''

import click
import sys
import spikeglx
from pathlib import Path
import numpy as np
import re
import pandas as pd  
import neurodsp.utils



# TODO: compute the power from a calibration 
# TODO: add option to extract the binary 
# TODO:SYNCHRONIZE
# TODO output in alf format (intervals, amplitudes)
# Just copy-pasted some functions from my module for ease of import

def binary_onsets(x,thresh):
    '''
    Get the onset and offset samples of a binary signal 
    :param x: signal
    :param thresh: Threshold
    :return: ons,offs
    '''
    xbool = x>thresh

    ons = np.where(np.diff(xbool.astype('int'))==1)[0]
    offs = np.where(np.diff(xbool.astype('int'))==-1)[0]
    if xbool[0]:
        offs = offs[1:]
    if xbool[-1]:
        ons = ons[:-1]
    if len(ons)!=len(offs):
        plt.plot(x)
        plt.axhline(thresh)
        raise ValueError('Onsets does not match offsets')
    return(ons,offs)


def get_opto_df(raw_opto,v_thresh,ni_sr,min_dur=0.001,max_dur=10):
    '''
    :param raw_opto: raw current sent to the laser or LED (1V/A)
    :param v_thresh: voltage threshold to find crossing
    :param ni_sr: sample rate (Hz)
    :param min_dur: minimum stim duration in seconds
    :param max_dur: maximum stim duration in seconds
    :return: opto-df a dataframe with on, off, and amplitude
    '''
    ons,offs = binary_onsets(raw_opto,v_thresh)
    durs = offs-ons
    opto_df = pd.DataFrame()
    opto_df['on'] = ons
    opto_df['off'] = offs
    opto_df['durs'] = durs

    min_samp = ni_sr*min_dur
    max_samp = ni_sr*max_dur
    opto_df = opto_df.query('durs<=@max_samp & durs>=@min_samp').reset_index(drop=True)

    amps = np.zeros(opto_df.shape[0])
    for k,v in opto_df.iterrows():
        amps[k] = np.median(raw_opto[v['on']:v['off']])
    opto_df['amps'] = np.round(amps,2)

    opto_df['on_sec'] = opto_df['on']/ni_sr
    opto_df['off_sec'] = opto_df['off']/ni_sr
    opto_df['dur_sec'] = np.round(opto_df['durs']/ni_sr,3)

    return(opto_df)

def process_rec(SR,opto_chan=5,v_thresh=0.5,**kwargs):
    '''
    Create a dataframe where each row is an opto pulse. 
    Information about the pulse timing, amplitude, and duration are created.
    '''

    
    print('\tReading raw data...',end='')
    raw_opto = SR.read(nsel=slice(None,None,None),csel=opto_chan)[0]
    print('done')
    df = get_opto_df(raw_opto,v_thresh,SR.fs,**kwargs)
    return(df)


def get_sync(SR_ni,SR_imec):
    """Find the relationship between the NI and the Imec timestamps.

    Args:
        SR_ni (_type_): _description_
        SR_imec (_type_): _description_
    """    
    ni_sync_signal = SR_ni.read_sync_digital(_slice=slice(None,None))[:,7]
    imec_sync_signal = SR_imec.read_sync_digital(_slice=slice(None,None))[:,6]
    ni_ts = np.where(np.diff(ni_sync_signal)==1)[0] / SR_ni.fs
    imec_ts = np.where(np.diff(imec_sync_signal)==1)[0] / SR_imec.fs
    fcn_ni2imec0,drift_ppm = neurodsp.utils.sync_timestamps(ni_ts,imec_ts)
    print(f'Timestamp drift of {drift_ppm}')
    return(fcn_ni2imec0)
    
def apply_sync_to_df(fcn,df):
    df['on_sec_corrected'] = fcn(df['on_sec'])
    df['off_sec_corrected'] = fcn(df['off_sec'])
    return(df)


@click.command()
@click.argument('gate_path' )
@click.option('--opto_chan','-o',default=5,help = 'Analog channel of the optogenetic pulses',show_default=True)
@click.option('--v_thresh','-v',default=0.6,help = 'voltage threshold to register a pulse',show_default=True)
def main(gate_path,opto_chan,v_thresh):
    opto_chan = int(opto_chan)
    v_thresh = float(v_thresh)

    #TODO: Make general to operate on a run
    #TODO: Make general for arbitrary trials

    
    gate_path = Path(gate_path)
    ni_list = list(gate_path.glob('*nidq.bin'))
    ni_list.sort()
    imec_fn = list(gate_path.rglob('*.ap.bin'))[0]
    # print(ni_list)
    for ni_fn in ni_list:
        print(f'Processing {ni_fn}',end='...')
        trig_string = re.search('t\d{1,3}',ni_fn.stem).group()
        SR_ni = spikeglx.Reader(ni_fn)
        SR_imec = spikeglx.Reader(imec_fn)
        df = process_rec(SR_ni,opto_chan=opto_chan,v_thresh=v_thresh)
        fcn_ni2imec0 = get_sync(SR_ni,SR_imec)
        df = apply_sync_to_df(fcn_ni2imec0,df)

        df.to_csv(ni_fn.parent.joinpath(f'optostims.{trig_string}.times.tsv'),sep='\t')
        print('done.')

        
    print('Done with all!')

if __name__ == '__main__':
    main()




