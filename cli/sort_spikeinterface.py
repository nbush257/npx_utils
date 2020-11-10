import spikeinterface.extractors as se
import click
import spikeinterface.sorters as ss
import sys
from pathlib2 import Path
sys.path.append('../')
sys.path.append('./')
from readSGLX import *
from SGLXMetaToCoords import MetaToCoords
import os
import pandas as pd
import json
from ks2_cleaning_pipeline import run_post_sort_pipeline

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
        output_dir = os.path.join(p_load,'ks2')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    meta_fn = os.path.splitext(fn)[0] + '.meta'
    prb_fn = meta2prb(meta_fn)
    rec = se.SpikeGLXRecordingExtractor(fn)
    rec = se.load_probe_file(rec, prb_fn)
    params = ss.get_default_params('kilosort2')
    #params['minFR'] = 0.2
    #params['minfr_goodchannels'] = 0.2
    # params['car'] = False
    # params['NT'] = 64*512 + params['ntbuff']
    print('='*50)
    print('Running Kilosort2')
    print('='*50)
    print(f'\tInput data:\t{fn}')
    print(f'\tWriting to:\t{output_dir}')
    ks2 = ss.run_kilosort2(rec, **params, output_folder=output_dir)
    print('Done sorting. Running cleaning pipeline')
    run_post_sort_pipeline(output_dir,fn)


if __name__=='__main__':
    main()
