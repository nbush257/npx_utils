import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import sys
if sys.platform == 'linux':
    sys.path.append('/active/ramirez_j/ramirezlab/nbush/helpers/NeuropixelsRegistration/python')
    sys.path.append('/active/ramirez_j/ramirezlab/nbush/helpers/')
else:
    sys.path.append('Y:/helpers/NeuropixelsRegistration/python')
    sys.path.append('Y:/helpers/')
import estimate_displacement as ed
from numpy.fft import fft2, ifft2, fftshift, ifftshift # Python DFT
import pywt
from ibllib.io import spikeglx
from scipy.ndimage import gaussian_filter1d
from NeuropixelsRegistration.python import utils
from utils import mat2npy
import scipy.io.matlab as sio
import click
import glob
import shutil
import subprocess

def sanitize_meta(bin_fn):
    '''
    This removes any lines in the .meta file that have more than one "="
    It first dupliceates the meta file and inserts a ".ecephy" tag in the filename
    Then it removes the original .meta file
    Then it reads all the lines in the ".ecephys" meta file and writes all
        lines with only one "=" to the untagged meta file (will look like the original meta file).

    NB: IT IS UNCLEAR IF THIS WILL BREAK ECEPHYS.
        If is does, make a new function that deletes the meta file created here, and reinstates the .ecephys file as
        the standard by removing the tag.
    :param bin_fn:
    :return:
    '''
    meta_fn = bin_fn.replace('.bin','.meta')
    new_meta_fn = meta_fn[:-4] + 'ecephys.' + meta_fn[-4:]
    shutil.copy(meta_fn,new_meta_fn)
    os.remove(meta_fn)
    with open(new_meta_fn,'r') as fid:
        for ll in fid:
            dum = ll.split('=')
            if len(dum)>2:
                continue
            else:
                with open(meta_fn,'a') as fid2:
                    fid2.write(ll)

def create_registered_meta(bin_fn,registered_fn):
    '''
    Copies the meta data so that the registered binary data can be read
    by spike glx.
    #TODO: overwrite some file parameters to be able to be read by spikeglx. Untested currently. Probably helpful for keeping things like samplerate
    :param bin_fn:
    :return:
    '''
    meta_fn = bin_fn.replace('.bin', '.meta')
    new_meta_fn = meta_fn.replace('tcat.imec','tcat.registered.imec')
    b = os.path.getsize(registered_fn)
    with open(meta_fn,'r') as fid:
        for ll in fid:
            dum = ll.split('=')
            if dum[0] == 'fileSizeBytes':
                with open(new_meta_fn, 'a') as fid2:
                    fid2.write(f'fileSizeBytes={b}\n')
            else:
                with open(new_meta_fn,'a') as fid2:
                    fid2.write(ll)

def plot_shift(total_shift,probe_dir):
    '''
    Plots the total shift and driftmap
    :param total_shift:
    :param probe_dir:
    :return:
    '''
    f,ax = plt.subplots(nrows=2,sharex=True,figsize=(6,5))
    plt.sca(ax[0])
    h=ax[0].pcolormesh(total_shift,cmap='RdBu_r',vmin=-50,vmax=50)

    ax[0].set_ylabel('Channel Pos (um)')

    plt.colorbar(h)
    ax[1].set_xlabel('time(s)')

    ax[1].plot(total_shift[::300,:].T)
    ax[1].set_ylabel('probe shift (um)')
    plt.sca(ax[1])
    plt.colorbar(h)
    plt.savefig(os.path.join(probe_dir,'driftmat_decentralized.png'),dpi=300)
    plt.close('all')



@click.command()
@click.argument('probe_dir')
@click.option('--resume','-r',is_flag=True)
@click.option('--iteration_num','-i',default=4)
@click.option('--batch_num','-b',default=0)
@click.option('--compute_shift','-c',is_flag=True)
def main(probe_dir,resume,iteration_num,batch_num,compute_shift):
    chan_map_fn = glob.glob(os.path.join(probe_dir,'*ap*chanMap.mat'))[0]
    bin_fn = glob.glob(os.path.join(probe_dir,'*tcat.imec*ap.bin'))[0]
    print(f"Registering\n\t{bin_fn}")
    if sys.platform=='linux':
        registered_dir = os.path.join(probe_dir,'registered')
    else:
        registered_dir = 'D:/registered'




    sanitize_meta(bin_fn)
    geomarray = mat2npy(chan_map_fn)
    reader = spikeglx.Reader(bin_fn)

    shift_fn = os.path.join(probe_dir,'total_shift.npy')
    # Load in the total shift file if it exists, and we do not explicitly as to compute it.
    if os.path.exists(shift_fn) and not compute_shift:
        print('LOADING IN TOTAL SHIFT FILE')
        total_shift = np.load(shift_fn)
    else:
        print("Computing total_shift")
        total_shift = ed.estimate_displacement(reader, geomarray,
                                               reader_type='spikeglx',
                                               resume_with_raster=resume,
                                               working_directory=probe_dir,
                                               iteration_num=iteration_num)
    plot_shift(total_shift,probe_dir)

    if batch_num == 0:
        n_batches = int(np.floor(reader.ns/reader.fs))
    else:
        n_batches = batch_num
    if not os.path.exists(registered_dir):
        os.makedirs(registered_dir)

    print('Registering...')
    ed.register_data(reader,total_shift,geomarray,registered_dir,interp='linear',reader_type='spikeglx',n_batches=n_batches)
    print('Registered!')
    output_name = os.path.split(bin_fn)[1]
    output_name = output_name.replace('tcat.imec','tcat.registered.imec')
    print('Merging...')

    if not sys.platform == 'linux':
        utils.merge_filtered_files(registered_dir, 'D:/', output_name, delete=False)
        print('\n')
        print('Moving data to probe_dir')
        print('\n')
        src = fr'D:\{output_name}'
        subprocess.call(['xcopy', src, probe_dir,'/y'])
    else:
        utils.merge_filtered_files(registered_dir, probe_dir, output_name, delete=False)
    shutil.rmtree(registered_dir)

    print('Merged!')
    sio.savemat(os.path.join(probe_dir,'mc_meta.mat'),{'sr':reader.fs,'chanMap':geomarray,'n_chans':geomarray.shape[0],'total_shift':total_shift})
    create_registered_meta(bin_fn,os.path.join(probe_dir,output_name))
if __name__=='__main__':
    main()