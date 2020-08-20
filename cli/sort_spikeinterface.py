import spikeinterface.extractors as se
import click
import spikeinterface.sorters as ss
import sys
from pathlib2 import Path
sys.path.append('../')
from readSGLX import *
from SGLXMetaToCoords import MetaToCoords
import os
import pandas as pd
import json

def meta2prb(meta_fn):
    MetaToCoords(Path(meta_fn),0,showPlot=False)
    coord_fn = meta_fn.replace('.meta','_siteCoords.txt')
    prb_fn = meta_fn.replace('.meta','_siteCoords.prb')
    coords = pd.read_csv(coord_fn,delimiter='\t',header=None)
    coords[[1,2]].values;
    coord_dict = {0:{
        'channels':coords[0].values.tolist(),
        'geometry':coords[[1,2]].values.tolist()
                    }}
    with open(prb_fn,'w') as fid:
        fid.write('channel_groups = ')
        json.dump(coord_dict,fid)
    return(prb_fn)

@click.command()
@click.argument('fn')
@click.option('-o','--output_dir',type=str,default=None)
def main(fn,output_dir):
    if output_dir is None:
        p_load = '/'.join(Path(fn).parts[:-1])
        output_dir = p_load.replace('raw','processed')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    meta_fn = os.path.splitext(fn)[0] + '.meta'
    prb_fn = meta2prb(meta_fn)
    rec = se.SpikeGLXRecordingExtractor(fn)
    rec = se.load_probe_file(rec, prb_fn)
    params = ss.get_default_params('kilosort2')
    params['minFR'] = 1
    params['minfr_goodchannels'] = 0
    print('='*50)
    print('Running Kilosort2')
    print('='*50)
    print(f'\tInput data:\t{fn}')
    print(f'\tWriting to:\t{output_dir}')
    ks2 = ss.run_kilosort2(rec, **params, output_folder=output_dir)
    print('Done!')

if __name__=='__main__':
    main()

