"""Routines for data import and manipulation."""
import os
from pynwb import NWBFile,TimeSeries,NWBHDF5IO
from pynwb.ecephys import *
import click
import tables
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
sys.path.append('./src')
import datetime
from dateutil.tz import tzlocal
from pypl2 import pl2_ad, pl2_spikes, pl2_events, pl2_info
from pypl2.pypl2lib import *
import spikeinterface as si
import spikeinterface.widgets as sw
import spikeinterface.extractors as se
import spikeinterface.sorters as ss

def ad2df_int16(pl2_file,ad_spec='SPKC'):
    '''
    return a dataframe with int16 values - save space and conversions
    :param pl2_file:
    :param ad_spec:
    :return:
    '''
    spkinfo,evtinfo,adinfo = pl2_info(pl2_file)

    # Find desired data that is not empty
    fx = lambda x,ad_spec : ad_spec in x.name
    fx2 = lambda x : x.n!=0
    use_chans = np.where([fx(x,ad_spec) for x in adinfo])[0]
    has_data = np.where([fx2(x) for x in adinfo])[0]
    use_chans = np.intersect1d(use_chans,has_data)

    # Initialize
    ncols = len(use_chans)
    nrows = adinfo[use_chans[0]].n
    dat = np.empty([nrows,ncols],dtype='int16')
    names = []

    p = PyPL2FileReader()
    handle = p.pl2_open_file(pl2_file)
    file_info = PL2FileInfo()
    res = p.pl2_get_file_info(handle, file_info)

    # Extract
    for ii,chan in enumerate(tqdm(use_chans)):

        achannel_info = PL2AnalogChannelInfo()
        res = p.pl2_get_analog_channel_info(handle, int(chan), achannel_info)
        num_fragments_returned = c_ulonglong(0)
        num_data_points_returned = c_ulonglong(0)
        fragment_timestamps = (c_longlong * achannel_info.m_MaximumNumberOfFragments)()
        fragment_counts = (c_ulonglong * achannel_info.m_MaximumNumberOfFragments)()
        values = (c_short * achannel_info.m_NumberOfValues)()
        names.append(adinfo[chan].name)

        res = p.pl2_get_analog_channel_data(handle,
                                            int(chan),
                                            num_fragments_returned,
                                            num_data_points_returned,
                                            fragment_timestamps,
                                            fragment_counts,
                                            values)
        temp = np.array(values)


        dat[:,ii] = temp

    sr = achannel_info.m_SamplesPerSecond
    ts = np.arange(0,nrows/sr,1/sr)
    gain = achannel_info.m_CoeffToConvertToUnits
    df = pd.DataFrame(data=dat,index=ts,columns=names)
    df.index.name='Time'
    p.pl2_close_file(handle)
    return(df,gain)


def ad2df(pl2_file,ad_spec='SPKC',convert2int=False):
    '''
    Get analog data from a plexon file and store as a user friendly dataframe
    :param pl2_file: path to pl2 file
    :param ad_spec: specify the type of data to extract ['WB','SPKC','AI','FP']
    :return: dataframe with extracted channels
    '''
    # Get basic file info
    spkinfo,evtinfo,adinfo = pl2_info(pl2_file)

    # Find desired data that is not empty
    fx = lambda x,ad_spec : ad_spec in x.name
    fx2 = lambda x : x.n!=0
    use_chans = np.where([fx(x,ad_spec) for x in adinfo])[0]
    has_data = np.where([fx2(x) for x in adinfo])[0]
    use_chans = np.intersect1d(use_chans,has_data)

    # Initialize
    ncols = len(use_chans)
    nrows = adinfo[use_chans[0]].n
    dat = np.empty([nrows,ncols],dtype='float32')
    names = []

    # Extract
    for ii,chan in enumerate(tqdm(use_chans)):
        temp_dat = pl2_ad(pl2_file,adinfo[chan].name)
        names.append(adinfo[chan].name)
        dat[:, ii] = temp_dat.ad

    # Get timestamps using last channels datarate
    ts = np.arange(0,nrows/temp_dat[0],1/temp_dat[0])

    # Package
    df = pd.DataFrame(data=dat,index=ts,columns=names)
    df.index.name='Time'

    return(df)


def ad2h5(pl2_file,ad_spec='SPKC'):
    '''
    Write the analog data to an h5 file
    :param pl2_file:
    :param ad_spec:
    :return:
    '''
    spkinfo,evtinfo,adinfo = pl2_info(pl2_file)
    save_file = os.path.splitext(pl2_file)[0]+'.h5'
    fx = lambda x,ad_spec : ad_spec in x.name
    use_chans = np.where([fx(x,ad_spec) for x in adinfo])[0]
    nrows = len(use_chans)
    ncols = adinfo[use_chans[0]].n


    f = tables.open_file(save_file, mode='w')
    atom = tables.Float32Atom()
    array_c = f.create_earray(f.root, 'data', atom, (0, ncols))
    for idx in use_chans:
        chan = adinfo[idx]
        if chan.n != ncols:
            raise(ValueError(f'Number of samples in channel {idx}:{chan.n} does not match first channel {ncols}'))
        chan_dat = np.array(pl2_ad(pl2_file,chan.name).ad,dtype='float32')
        array_c.append(chan_dat[np.newaxis,:])
    f.close()
    return(0)


def convert2int16(x):
    x = x/np.max(x)/2*2**16
    x = x.astype('int16')
    return(x)

def extract_rec_start(pl2_file):
    '''
    Extract the recording start time as a datetime object
    :return:
    '''
    p = PyPL2FileReader()
    handle = p.pl2_open_file(pl2_file)
    file_info = PL2FileInfo()
    res = p.pl2_get_file_info(handle, file_info)

    tm = file_info.m_CreatorDateTime
    dt = datetime.datetime(tm.tm_year-100+2000,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min,tm.tm_sec,tzinfo=tzlocal)
    p.pl2_close_file(handle)
    return(dt)


def extract_rec_duration(pl2_file):
    '''
    Return the recording duration in seconds and samplerate in samples per second
    :param pl2_file:
    :return: dur,sr
    '''
    p = PyPL2FileReader()
    handle = p.pl2_open_file(pl2_file)
    file_info = PL2FileInfo()
    res = p.pl2_get_file_info(handle, file_info)
    dur = file_info.m_DurationOfRecording/file_info.m_TimestampFrequency
    sr = file_info.m_TimestampFrequency
    p.pl2_close_file(handle)

    return(dur,sr)


def convert2NWB(pl2_file):
    '''
    This function specifically converts a plexon file to an NWB file. Does not
    concatenate across runs. Does not contain location metadata.
    Assumes a lot about the structure of the PL2 file
    :param pl2_file:
    :return:
    '''
    # Get file ID
    basename = os.path.split(os.path.splitext(pl2_file)[0])[1]
    savefile = os.path.splitext(pl2_file)[0]+'.nwb'
    savefile = savefile.replace('rawdat-','')

    SPKC = ad2df(pl2_file,'SPKC')
    nasal = ad2df(pl2_file,'AI')

    dt = extract_rec_start(pl2_file)
    dur,sr = extract_rec_duration(pl2_file)

    nwbfile = NWBFile('Plexon 2 Probe Recording with Nasal',basename,dt,
                      experimenter='Nick Bush',
                      lab = 'Ramirez Lab',
                      institution = 'CIBR'
                      )
    device = nwbfile.create_device(name='Plexon DHP')

    prb0 = nwbfile.create_electrode_group('A16-0',description='16channel neuronexus polytrode probe',
                                          location='Brainstem',
                                          device=device)
    prb1 = nwbfile.create_electrode_group('A16-1',description='16channel neuronexus polytrode probe',
                                          location='PAC',
                                          device=device)
    for ii in range(16):
        nwbfile.add_electrode(id=ii,location='',filtering='300-Inf',group = prb0,
                              x=-1.,y=-1.,z=-1.,imp=-1.)
    for ii in range(16,32):
        nwbfile.add_electrode(id=ii,location='',filtering='300-Inf',group = prb1,
                              x=-1.,y=-1.,z=-1.,imp=-1.)

    etr0 = nwbfile.create_electrode_table_region(list(range(16)),'prb0')
    etr1 = nwbfile.create_electrode_table_region(list(range(16)),'prb1')

    for ii,(col,data) in enumerate(SPKC.iteritems()):
        if ii==0:
            ephys_ts0 = ElectricalSeries(col, data.values, etr0, timestamps=SPKC.index.values)
            nwbfile.add_acquisition(ephys_ts0)
        elif ii<16:
            ephys_ts = ElectricalSeries(col,data.values,etr0,timestamps=ephys_ts0)
            nwbfile.add_acquisition(ephys_ts)
        else:
            ephys_ts = ElectricalSeries(col,data.values,etr1,timestamps=ephys_ts0)
            nwbfile.add_acquisition(ephys_ts)

    # add nasal thermistor
    aux = TimeSeries('nasal', nasal.values,timestamps=nasal.index.values,unit='v')
    nwbfile.add_acquisition(aux)

    print(f'Writing NWB file to {savefile}')
    with NWBHDF5IO(savefile,'w') as io:
        io.write(nwbfile)
    print('Done!')

