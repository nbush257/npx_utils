import click
import spikeglx
from pathlib import Path
import numpy as np
import re

def process_rec(ni_fn,trig_chan=6,verbose=True):
    SR = spikeglx.Reader(ni_fn)
    trig = SR.read_sync_digital(_slice=slice(None,None))[:,trig_chan]
    frame_samps = np.where(np.diff(trig)>0)[0]-1
    frame_times = frame_samps/SR.fs
    n_frames = frame_samps.shape[0]
    framerate = np.mean(1/np.diff(frame_times))
    framerate_std = np.std(1/np.diff(frame_times))
    framerate_range = (np.min(1/np.diff(frame_times)),np.max(1/np.diff(frame_times)))
    if verbose:
        print(f'Found {n_frames} frames')
        print(f'Mean frame rate of {framerate:0.2f} fps')
        print(f'S.D. frame rate of {framerate_std:0.2f} fps')
        print(f'Framerate min:{framerate_range[0]:0.2f}fps\tmax:{framerate_range[1]:0.2f}fps')
    return(frame_samps,frame_times)


@click.command()
@click.argument('gate_path' )
@click.option('--trig_chan','-c',default=6,type=int,help = 'Digital channel of the frame_trigger',show_default=True)
def main(gate_path,trig_chan):
    
    gate_path = Path(gate_path)
    ni_list = list(gate_path.glob('*nidq.bin'))
    ni_list.sort()
    
    for ni_fn in ni_list:
        print(f'Processing {ni_fn}',end='...')
        trig_string = re.search('t\d{1,3}',ni_fn.stem).group()
        frame_samps,frame_times = process_rec(ni_fn,trig_chan=trig_chan)
        np.save(gate_path.joinpath(f'frames.{trig_string}.times.npy'),frame_times)
        print('done.')
        
    print('Done with all!')

if __name__ == '__main__':
    main()

