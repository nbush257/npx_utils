'''
Extract digital signals from the NI and IMEC Streams
'''
import spikeglx
from pathlib import Path
import numpy as np
import spikeinterface.full as si
from ibllib.ephys import sync_probes
from ibllib.io.extractors import ephys_fpga
import ibldsp.utils 
import one.alf.io as alfio
import click
import logging

# Channels are hardcoded for the digital line using spike interface.
# We are using spikeinterface to read the raw data files because the IBL 
# Code cannot handle the commercial NP2 probes (2013) at this time
IMEC_CHAN = '#SY0'
NI_CHAN = '#XD0'
def _extract_sync(rec,stream,chan):
    dig = rec.get_traces(channel_ids = [stream+chan])
    dig = spikeglx.split_sync(dig)
    ind, polarities = ibldsp.utils.fronts(dig,axis=0)
    samps,chans = ind
    sync = {'times': samps/rec.sampling_frequency,
            'channels': chans,
            'polarities': polarities}
    return(sync)


@click.command()
@click.argument('session_path')
def main(session_path):
    ephys_files = spikeglx.glob_ephys_files(session_path)
    for efi in ephys_files:
        if 'nidq' in efi.keys():
            stream = 'nidq'
            chan=NI_CHAN
        else:
            stream = efi.get('label')[-5:]+'.ap'
            chan=IMEC_CHAN
        logging.info(f'Working on {efi.label}. Stream:{stream} Chan:{chan}')

        rec = si.read_spikeglx(efi['path'], stream_name=stream, load_sync_channel=True)
        sync = _extract_sync(rec,stream,chan)
        out_files = alfio.save_object_npy(efi['path'], sync, 'sync',
                                namespace='spikeglx', parts='')
        
        
        for x in out_files:
            logging.info(f'Saved \t{str(x)}')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()