'''
The recorded auxiliary data (e.g. diaphragm and pleth) is sampled at 10K.
While the high sampling rate is critical to acquire good EMG, it is excessive for both
the integrated and the pleth.

This script does the following:
1) Downsamples the plethysmography by 10x to be input into breathmetrics (BM does filtering and processing)
2) Filters the EMG (300-5K)
3) Integrates the EMG with a triangular window
4) Downsamples the integrated EMG by 10x to match the pleth
5) Extracts features from the integrated EMG.
6) Saves a .mat file with the downsampled integrated EMG and the Pleth
7) Saves a .csv with the diaphragm features
'''
import os
import re
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../utils')
import readSGLX
import numpy as np
import scipy.signal as sig
import scipy.io.matlab as sio
import proc
from pathlib import Path
import click
from sklearn.mixture import BayesianGaussianMixture
from scipy.ndimage.filters import median_filter


def remove_EKG(x,sr,,thresh=5):
    '''
    :param x: diaphragm (or other) emg
    :param sr: sample rate
    :return: y ekg filtered out
    '''
    sos = sig.butter(2,[5/sr/2,500/sr/2],btype='bandpass',output='sos')
    xs = sig.sosfilt(sos,x)
    pks = sig.find_peaks(xs,prominence=thresh*np.std(xs),distance=0.05*sr)[0]
    amps = xs[pks]
    win = int(0.010 *sr)
    y = x.copy()
    ekg = np.zeros([2*win,len(pks)])
    for ii,pk in enumerate(pks):
        try:
            ekg[:,ii] = x[pk-win:pk+win]
        except:
            pass

    ekg_std = np.std(ekg[:30],0) +np.std(ekg[-30:],0)
    ekg_std = np.log(ekg_std)
    bgm = BayesianGaussianMixture(2)
    cls = bgm.fit_predict(ekg_std[:,np.newaxis])
    cls[cls==0]=-1
    m0 = np.nanmean(ekg_std[cls==-1])
    m1 = np.nanmean(ekg_std[cls==1])
    if m0>m1:
        cls = -cls

    ww = int(.0005 * sr)
    ww += ww % 2 - 1
    for ii,pk in enumerate(pks):
        if pk-win<0:
            continue
        if (pk+win)>len(y):
            continue
        if cls[ii]==-1:
            sm_ekg = sig.savgol_filter(ekg[:,ii],ww,1)
            y[pk - win:pk + win] -= sm_ekg
        else:
            med_ekg = np.nanmedian(ekg[:,ii-5:ii+5],1)
            med_amp = np.median(amps[ii-5:ii+5])
            scl = amps[ii]/med_amp
            y[pk - win:pk + win] -=med_ekg*scl

    y[np.isnan(y)] = np.nanmedian(y)
    return(y)


def load_mmap(fn):
    '''
    Load a memory map of the auxiliary channels
    :param fn: filename to nidaq .bin file
    :return:
            mmap - memory map to the aux data
            sr - sampling rate of aux data
    '''
    meta = readSGLX.readMeta(Path(fn))
    mmap = readSGLX.makeMemMapRaw(fn,meta)

    return(mmap,meta)


def load_ds_pleth(mmap,meta,chan_id,ds_factor=10):
    '''
    Load and downsample the pleth data
    :param mmap: mmap
    :param meta: metadata dict
    :param chan_id: Pleth channel index
    :param ds_factor: downsample factor
    :return:
            dat- downsampled pleth data
            sr_sub - new sampling rate
    '''
    assert(type(ds_factor) is int )
    bitvolts = readSGLX.Int2Volts(meta)
    sr = readSGLX.SampRate(meta)
    dat = mmap[chan_id,::ds_factor]*bitvolts
    sr_sub = sr/ds_factor

    return(dat,sr_sub)


def load_dia_emg(mmap,meta,chan_id):
    '''
    Read the raw diaphragm emg
    :param mmap: memory ampped aux data
    :param meta: meta dict
    :param chan_id: channel index of the diaphragm
    :return:
        dat - the raw diaphramg
        sr - the smapling rate of the diaphragm recording
    '''
    bitvolts = readSGLX.Int2Volts(meta)
    sr = readSGLX.SampRate(meta)
    dat = mmap[chan_id]*bitvolts
    dat = dat-np.mean(dat)
    return(dat,sr)


def filt_int_ds_dia(x,sr,ds_factor=10):
    '''
    Filter, integrate and downsample the diaphragm. Detect and summarize the diaphragm bursts
    Uses medfilt to smooth so it is a little slow, but it is worth it.
    :param x:
    :param sr:
    :param ds_factor:
    :return:
    '''
    assert(type(ds_factor) is int)

    #Remove the EKG artifact
    print('Removing the EKG...')
    dia_filt = remove_EKG(x,sr,thresh=2)
    dia_filt[np.isnan(dia_filt)] = np.nanmedian(dia_filt)


    # Filter for high frequency signal
    sos = sig.butter(2,[300/sr/2,5000/sr/2],btype='bandpass',output='sos')
    dia_filt = sig.sosfilt(sos,dia_filt)

    # Use medfilt to get the smoothed rectified EMG
    print('Smoothing the rectified trace...')

    dd = median_filter(np.abs(dia_filt),int(sr*.05)+1)
    # Smooth it out a little more
    dia_smooth = sig.savgol_filter(dd,window_length=int(0.01*sr)+1,polyorder=1)

    # Downsample because we don't need this at the original smapling rate
    dia_sub = dia_smooth[::ds_factor]
    sr_sub = sr/ds_factor

    # get the burst statistics
    dia_df = proc.burst_stats_dia(dia_sub,sr_sub,rel_height=0.95)

    # Normalize the integrated diaphragm to a z-score.
    dia_sub = dia_sub/np.std(dia_sub)
    print('Done processing diaphragm')

    return(dia_df,dia_sub,sr_sub)


def make_save_fn(fn,save_path,save_name='_aux_downsamp'):

    load_path,no_path = os.path.split(fn)
    prefix = no_path.replace('.nidq.bin','')
    save_fn = os.path.join(save_path,prefix+save_name+'.mat')
    return(save_fn,prefix)


def main(fn,pleth_chan,dia_chan,save_path):

    if save_path is None:
        save_path = os.path.split(fn)[0]


    mmap,meta = load_mmap(fn)
    raw_dia,sr_dia = load_dia_emg(mmap,meta,dia_chan)
    dia_df,dia_sub,sr_dia_sub = filt_int_ds_dia(raw_dia,sr_dia)
    if pleth_chan<0:
        pleth = []
        sr_pleth = sr_dia_sub
    else:
        pleth,sr_pleth = load_ds_pleth(mmap,meta,pleth_chan)
        pleth = pleth/np.std(pleth)



    #Make sure the subsampled dia and pleth have identical SR
    assert(sr_pleth == sr_dia_sub)
    t = np.arange(0, len(dia_sub)/sr_pleth, 1 / sr_pleth)
    t = t[:len(dia_sub)]
    # Save the downsampled data to a mat file
    data_dict = {
        'pleth':pleth,
        'dia':dia_sub,
        'sr':sr_pleth,
        't':t
    }
    save_fn,prefix = make_save_fn(fn,save_path)
    sio.savemat(save_fn,data_dict,oned_as='column')

    # Save the extracted diaphragm to a csv
    # But strip the data referenced to the 10K sampling
    dia_df.drop(['on_samp','off_samp','duration_samp','pk_samp'],axis=1,inplace=True)
    dia_df.to_csv(os.path.join(save_path,f'{prefix}_dia_stat.csv'))

    dia_df['on_sec'].to_csv(os.path.join(save_path,f'{prefix}_dia_onsets.csv'),index=False)


@click.command()
@click.argument('fn')
@click.option('-p','--pleth_chan','pleth_chan',default=0)
@click.option('-d','--dia_chan','dia_chan',default=1)
@click.option('-s','--save_path','save_path',default=None)
def batch(fn,pleth_chan,dia_chan,save_path):
    '''
    Set pleth chan to -1 if no pleth is recorded.
    :param fn:
    :param pleth_chan:
    :param dia_chan:
    :param save_path:
    :return:
    '''
    if os.path.isdir(fn):
        print('Running as batch\n')
        for root,dirs,files in os.walk(fn):
            r = re.compile('.*nid.*bin')
            flist = list(filter(r.match, files))
            if len(flist)>0:
                print(flist)
                for ff in flist:
                    fname = os.path.join(root,ff)
                    print(fname)
                    try:
                        main(fname,pleth_chan,dia_chan,root)
                        if pleth_chan>=0:
                            matlab_cmd_string = "matlab -r -nosplash -nodesktop -nojvm bm_mat_proc('" + fname + "')"
                            os.system(matlab_cmd_string)
                        else:
                            print('No pleth signal so not performing BM')

                    except:
                        print('='*50)
                        print(f'Failure on file {fname}')
                        print('='*50)
    else:
        root = os.path.split(fn)[0]
        main(fn, pleth_chan, dia_chan, root)
        if pleth_chan>=0:
            matlab_cmd_string = "matlab -r -nosplash -nodesktop -nojvm bm_mat_proc('" + fn + "')"
            os.system(matlab_cmd_string)
        else:
            print('No pleth signal so not performing BM')


if __name__=='__main__':
    batch()
