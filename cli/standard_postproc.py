import click
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import os
from spykes.plot import NeuroVis
from spykes.ml.neuropop import NeuroPop
import seaborn as sns
from matplotlib import cm
import pickle
from affinewarp import PiecewiseWarping
import sys
sys.path.append('../')

try:
    import npx_utils.proc as proc
    import npx_utils.models as models
    import npx_utils.data as data
    import npx_utils.visualize as viz
except:
    import proc
    import models
    import data
    import visualize as viz

def test():
    ks2_dir = r'C:\Users\nbush\Desktop\explor_data\ks2'
    ni_bin_fn = r'C:\Users\nbush\Desktop\explor_data\m2020-18_g0_t0.nidq.bin'
    DIA_CHAN=1
    PLETH_CHAN=0

    pass

@click.command()
@click.argument('ks2_dir')
@click.argument('ni_bin_fn')
def main(ks2_dir,ni_bin_fn,PLETH_CHAN=0,DIA_CHAN=1,OPTO_CHAN=2,p_save=None):

    # (DONE) Plot long time-scale raster of all neurons with respiratory rate
    # (DONE) Plot short timescale raster of all neurons with pleth and dia integrated
    # (DONE) Plot respiratory rate and heart rate
    # (DONE) Plot phase tuning and event psth of dia on for each neuron individually.
    #   Include gasp tuning (?) Include apnea tuning (?)
    # Plot event triggered raster of several example breaths
    # Plot raster/psth of opto-triggered respones.
    # Organize plots into a single image
    # Plot depth map of phase tuning.
    # Plot depth map of Tuning depth
    # Plot depth map of tagged neurons
    # Plot Dia/pleth and onsets to verify the detections
    # Compute tagged boolean for all neuron
    # Compute recording depth, preferred phase, modulation depth, and is-modulated for all neurons
    # Save Unit level data in csv

    # NEEDS: ks2dir,ni_binary, epoch times + labels, channels

    # ========================= #
    #  LOAD DATA #
    # ========================= #
    if p_save is None:
        p_save = os.path.join(ks2_dir,'..')
    # Need to process in line for memory reasons
    plt.style.use('seaborn-paper')
    sr = data.get_sr(ni_bin_fn)
    spikes = data.get_concatenated_spikes(ks2_dir)
    spikes = data.filter_by_spikerate(spikes,500)
    dia = data.get_ni_analog(ni_bin_fn,DIA_CHAN)
    dia_df,phys_df,dia_int = proc.proc_dia(dia,sr)
    pleth = data.get_ni_analog(ni_bin_fn,PLETH_CHAN)
    pleth_df = proc.proc_pleth(pleth,sr)
    tvec = np.arange(0,len(dia)/sr,1/sr)

    # Compute Phase
    b,a = scipy.signal.butter(4,25/sr/2,btype='low')
    pleth_f = scipy.signal.filtfilt(b,a,pleth)
    phi = proc.calc_phase(pleth_f)
    phi = proc.shift_phi(phi,pleth_df['on_samp'])

    # Plot raster summaries
    f,ax = viz.plot_raster_summary(spikes,phys_df)
    plt.savefig(os.path.join(p_save,'example_raster_long.png'),dpi=150)
    viz.plot_raster_example(spikes,sr,pleth,dia_int,500,5)
    plt.savefig(os.path.join(p_save,'example_raster_short.png'),dpi=150)

    # Plot single cell summaries
    epochs = np.arange(0,1501,300)
    for cell_id in spikes['cell_id'].unique():
        f = viz.plot_cell_summary_over_time(spikes,cell_id,epochs,dia_df,phi,sr)
        f.savefig(os.path.join(p_save,f'summary_cellid{cell_id}.png'),dpi=150)
        plt.close('all')

    # =====================
    # Attempt a spykes model
    # First, downsample inputs:
    # THIS TAKES FOREVER AND ISN'T CONVERGING 2020-12-22 NEB
    # =====================
    # binsize=  0.005
    # tstop = 300
    # tbase = np.arange(0,tstop,binsize)
    # idx = np.searchsorted(tvec,tbase)
    # X1 = dia_int[idx]
    # X2 = phi[idx]
    #
    # # Use b
    # Y,cells,tt = proc.bin_trains(spikes.ts,spikes.cell_id,max_time=tstop,binsize=binsize,start_time=0)
    # Y = Y.T
    # pop = NeuroPop(n_neurons=Y.shape[1],verbose=True)
    #
    # pop.fit(X1,Y)
    #

    # =====================
    # Run the tensor factorization without affine warping
    # =====================
    raster,cell_id,bins = proc.bin_trains(spikes.ts,spikes.cell_id,max_time=epochs[-1],binsize=.005)
    mask = np.sum(raster,1)>100
    raster = raster[mask,:]
    cell_id = cell_id[mask]
    dd = spikes.groupby('cell_id').mean()['depth']
    dd = dd.loc[np.where(mask)[0]].values
    TT,raster_bins =models.raster2tensor(raster,bins,dia_df['on_sec'],pre=0.25,post=0.5)

    # Plot the mean population traces aligned to breath
    mean_breath = np.mean(TT,2).T
    order = np.argsort(np.argmax(mean_breath,1))
    f=plt.figure(figsize=(2,5))
    plt.pcolormesh(raster_bins,np.arange(len(order)),mean_breath[order,:],cmap='Greys')
    plt.axvline(0,ls=':',color='r')
    plt.xlabel('time [s]')
    plt.ylabel('neurons')
    sns.despine()
    plt.tight_layout()

    # Fit Tensor decomp
    np.random.seed(42)
    mdl,axs = models.get_best_TCA(TT,max_rank=12,plot_tgl=True)
    plt.close('all')

    # plot tensor decomp
    factors = mdl.factors.factors
    viz.plot_tensortools_factors(mdl,raster_bins,dd)
    plt.savefig(os.path.join(p_save,'tensor_factors.png'),dpi=150)
    plt.close('all')

    with open(os.path.join(p_save,'tensor_decomp.pkl'),'wb') as fid:
        pickle.dump(factors,fid)

    # =====================
    # Try affine warping
    # =====================

    # This commented section is if you want to rebin the tensors.
    # raster,cell_id,bins = proc.bin_trains(spikes.ts,spikes.cell_id,max_time=epochs[-1],binsize=.025)
    # mask = np.sum(raster,1)>100
    # raster = raster[mask,:]
    # cell_id = cell_id[mask]
    # dd = spikes.groupby('cell_id').mean()['depth']
    # dd = dd.loc[np.where(mask)[0]].values
    # TT,raster_bins =models.raster2tensor(raster,bins,dia_df['on_sec'],pre=0.25,post=0.5)
    TTt = np.transpose(TT,[2,0,1])

    warp_mdl = PiecewiseWarping(n_knots=1)
    warp_mdl.fit(TTt,iterations=25)

    warped = warp_mdl.transform(TTt)
    warped = np.transpose(warped,[1,2,0])
    warp_decomp,axs = models.get_best_TCA(warped,max_rank=12,plot_tgl=True)
    f = viz.plot_tensortools_factors(warp_decomp,raster_bins,dd)

    # Save
    plt.savefig(os.path.join(p_save,'tensor_factors_warped.png'),dpi=150)
    plt.close('all')

    with open(os.path.join(p_save,'tensor_decomp_warped.pkl'),'wb') as fid:
        pickle.dump(factors,fid)


if __name__=='__main__':
    main()




















