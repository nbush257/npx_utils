''' 
Script to convert the probe tracks from the AP_histology package in matlab to the needed structure fort he IBL GUI to align ephys
2023-09-01
'''
# https://github.com/int-brain-lab/iblapps/wiki/4.-Preparing-data-for-ephys-GUI

import ibllib.atlas as atlas
import numpy as np
import json
import scipy.io.matlab as sio
from pathlib import Path
import click

# resolution of Allen CCF atlas
res = 10
brain_atlas = atlas.AllenAtlas(res)

def convert_from_AP_histology(fn):
    '''
    fn - File name of the probeCCF file output from AP_histology
    Writes xyz_json files to the same folder. You will have to  manually move them to the appropriate alf folder.
    
    '''
    # Transform coords from CCF origin (order apdvml) to Bregma origin (order mlapdv)
    print('Converting from AP_histology')
    fn = Path(fn)
    dat_in = sio.loadmat(fn)
    n_probes = dat_in['probe_ccf'].shape[0]
    for probe_ID in range(n_probes):
        ccf_apdvml = dat_in['probe_ccf'][probe_ID]['points'][0] *10     # Upscale by 10 because they come in as indices into the 10uM atlas.
        xyz_mlapdv = brain_atlas.ccf2xyz(ccf_apdvml, ccf_order='apdvml') * 1e6 # convert bregma relative in meters to bregma relative in uM
        xyz_picks = {'xyz_picks': xyz_mlapdv.tolist()}

        output_path = fn.parent
        with open(Path(output_path, f'xyz_picks{probe_ID}.json'), "w") as f:
            json.dump(xyz_picks, f, indent=2)

def convert_from_SHARPTRACK(fn):
    '''
    Given a probe_points.mat file find all the probe matrices and conver to IBL. WIll write an xyz_json in the same folder.
    You will need to manually move to the appropriate ALF folder
    '''
    print('Converting from sharptrack')
    fn = Path(fn)
    dat_in = sio.loadmat(fn)
    n_probes = dat_in['pointList'][0,0][0].shape[0]

    for probe_ID in range(n_probes):
        points = dat_in['pointList'][0,0][0][probe_ID,0]
        ccf_apdvml = points[:,[2,1,0]]*10
        # May need to sort
        xyz_mlapdv = brain_atlas.ccf2xyz(ccf_apdvml, ccf_order='apdvml') * 1e6 # convert bregma relative in meters to bregma relative in uM
        xyz_picks = {'xyz_picks': xyz_mlapdv.tolist()}

        output_path = fn.parent
        with open(Path(output_path, f'xyz_picks{probe_ID+1}.json'), "w") as f:
            json.dump(xyz_picks, f, indent=2)


@click.command()
@click.argument('fn')
@click.argument('mode')
def main(fn,mode):
    if mode=='AP':
        convert_from_AP_histology(fn)
    elif mode=='ST':
        convert_from_SHARPTRACK(fn)
    else:
        raise NotImplementedError('Mode is not implemented. Must be "AP" or "ST"')

if __name__=='__main__':
    main()


                                  