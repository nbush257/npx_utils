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
import glob
from tqdm import tqdm
import re
sys.path.append('../')
from sklearn.preprocessing import StandardScaler

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

def plot_epochs(breaths,epochs,aux):
    f = plt.figure(figsize=(7,6))
    ax = f.add_subplot(311)
    ax.plot(breaths['on_sec'],1/breaths['postBI'],'k.',alpha=0.05)
    ax2 = ax.twinx()
    ax2.plot(aux['t'],aux['dia'],'tab:red',lw=0.5,alpha=0.5)
    ymax = 1/breaths['postBI'].min()
    for ii,v in epochs.iterrows():
        ax.axvline(v['t0']*60,c='r',ls=':')
        if(np.isnan(v['tf'])):
            v['tf'] = breaths.iloc[-1]['off_sec']/60
        ax.text(v['t0']*60,ymax,v['label'])
    ax.set_ylabel('Frequency (Hz)')
    ax2.set_ylabel('Dia',color='tab:red')

    ax3 = f.add_subplot(312,sharex=ax,sharey=ax)
    ax3.plot(breaths['on_sec'],1/breaths['postBI'].rolling(50).mean(),'k',lw=1,alpha=0.7,label='mean_freq')
    ax4 = ax3.twinx()
    ax4.plot(breaths['on_sec'],breaths['postBI'].rolling(50).std(),'tab:green',lw=1,alpha=0.7,label='std_freq')
    for ii,v in epochs.iterrows():
        ax3.axvline(v['t0']*60,c='r',ls=':')
        if(np.isnan(v['tf'])):
            v['tf'] = breaths.iloc[-1]['off_sec']/60
        ax3.text(v['t0']*60,ymax,v['label'])
    ax3.set_ylabel('Frequency (Hz)')
    ax4.set_ylabel('IBI Std',color='tab:green')

    ax4.set_xlabel('Time(s)')

    ax5 = f.add_subplot(313)
    ax5.plot(breaths['on_sec'],breaths['inhale_volumes'],'k.',alpha=0.1)
    ax5.set_ylabel('Inhale Volume (a.u.)')
    for ii,v in epochs.iterrows():
        ax5.axvline(v['t0']*60,c='r',ls=':')
        if(np.isnan(v['tf'])):
            v['tf'] = breaths.iloc[-1]['off_sec']/60

    sns.despine(trim=True,right=False)
    plt.tight_layout()
    for aax in [ax,ax3,ax5]:
        aax.grid(axis='x')


def plot_long_raster(spikes,breaths,epochs):
    raster, cell_id, bins = proc.bin_trains(spikes.ts, spikes.cell_id, binsize=5, start_time=0, max_time=spikes.ts.iloc[-1])
    f = plt.figure(figsize=(4,7))
    gs = f.add_gridspec(7,1)
    ax1 = f.add_subplot(gs[:-2,0])
    # ax1.pcolormesh(bins,cell_id,raster,cmap='Greys',vmin=0,vmax=np.percentile(raster,85))
    ax1.pcolormesh(bins,cell_id,raster,cmap='Greys')
    ax2 = f.add_subplot(gs[-2:-1,0],sharex=ax1)
    ax2.plot(breaths['on_sec'],1/breaths['postBI'],'k.',alpha=0.05)
    ax2.set_ylim(0,np.percentile(1/breaths['postBI'].dropna(),99.9))

    ymax = np.max(cell_id)
    for ii,v in epochs.iterrows():
        ax1.axvline(v['t0']*60,c='r',ls=':')
        ax2.axvline(v['t0']*60,c='r',ls=':')
        if(np.isnan(v['tf'])):
            v['tf'] = breaths.iloc[-1]['off_sec']/60
        ax1.text(v['t0']*60,ymax,v['label'],rotation=45)
    sns.despine()
    ax1.set_ylabel('Neurons')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Freq. (Hz)')
    plt.tight_layout()


def plot_single_cell_summary(spikes, neuron_id, epoch_t, dia_df, phi, sr, opto_times):
    '''
    assumes continuous epochs
    :param spikes:
    :param neuron_id:
    :param epoch_t:
    :param dia_df:
    :param phi:
    :param sr:
    :return:
    '''

    # Sample data
    ts = spikes[spikes.cell_id==neuron_id].ts.values
    if ts.shape[0]==0:
        return(0)
    depth = spikes[spikes.cell_id==neuron_id].depth.mean()
    epochs_t0 = epoch_t[:-1]
    epochs_tf = epoch_t[1:]
    cat = proc.events_in_epochs(dia_df['on_sec'], epoch_t)
    dia_df['cat'] = cat
    neuron = NeuroVis(ts)

    # Start plotting
    f = plt.figure(figsize=(7,7))
    gs = f.add_gridspec(nrows=4,ncols=6)
    ax0 = f.add_subplot(gs[0,:2],projection='polar')

    # Plot the polar rsponses
    cmap = cm.copper(np.linspace(0, 1, len(epoch_t)))
    KL = []

    for ii, (t0, tf) in enumerate(zip(epochs_t0, epochs_tf)):
        sub_spikes = ts[ts > t0]
        sub_spikes = sub_spikes[sub_spikes < tf]

        t0_samp = int(t0 * sr)
        tf_samp = int(tf * sr)
        phi_slice = phi[t0_samp:tf_samp]
        btrain = np.zeros_like(phi_slice)

        spt = sub_spikes - t0
        sp_idx = np.round(spt * sr).astype('int')-1
        btrain[sp_idx] = 1
        rr, theta_k, theta, L_dir = proc.angular_response_hist(phi_slice, btrain, 25)
        kl = proc.compute_KL(phi_slice,btrain,25)
        KL.append(kl)
        rr = np.hstack([rr,rr[0]])
        plt.polar(theta_k, rr * sr, color=cmap[ii])
    plt.xticks([0,np.pi/2,np.pi,3*np.pi/2])

    plt.suptitle(f'Neuron {neuron_id}; loc:{depth:0.0f}um; mod_depth:{np.mean(KL):0.2f}')

    # Get diaphragm triggered
    ax1 = f.add_subplot(gs[1,:2])
    # This line is needed to throw out all time after last epoch, which is gnerally longer thean all specified epochs
    dia_df = dia_df[dia_df['cat']<dia_df.cat.max()]
    psth_dia_off = neuron.get_psth(df=dia_df, event='on_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=20)
    plt.title(f'Dia On')
    legend_vals = ['Dia On']
    legend_vals += [f'{x/60}-{y/60}m' for x, y in zip(epochs_t0, epochs_tf)]
    leg = plt.legend([])
    ax1.get_legend().remove()

    ax2 = f.add_subplot(gs[2,:2],sharex=ax1,sharey=ax1)
    psth_dia_off = neuron.get_psth(df=dia_df, event='off_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=20)
    plt.title(f'Dia Off')
    plt.ylabel('')

    legend_vals = ['Trigger']
    legend_vals += [f'{x}s - {y}s' for x, y in zip(epochs_t0, epochs_tf)]
    plt.legend(legend_vals,bbox_to_anchor=(1.2,0.5),loc='center right',fontsize=6)

    ax3 = f.add_subplot(gs[:,2:4])
    raster = neuron.get_raster(df=dia_df,event='on_sec',window=[-500,500],binsize=5,plot=False)
    trial,tspike = np.where(raster['data'][0])
    bins = np.arange(raster['window'][0],raster['window'][1],raster['binsize'])
    ax3.plot(bins[tspike],dia_df['on_sec'][trial]/60,'k.',ms=2,alpha=0.1)
    ax3.set_ylabel('time [min]')
    ax3.set_xlabel('time [ms]')
    ax3.axvline(0,color='r',ls=':')
    ax3.set_ylim(epoch_t[0] / 60, epoch_t[-1] / 60)
    for ii, (t0, tf) in enumerate(zip(epochs_t0, epochs_tf)):
        ax3.axhspan(t0/60,tf/60,color=cmap[ii],alpha=0.1)
        ax3.axhline(t0/60,color='k',ls=':')

    ax4 = f.add_subplot(gs[:,-2:],sharey=ax3)
    ax4.plot(1/dia_df['postBI'],dia_df['on_sec']/60,'.',color='tab:red',alpha=0.3)

    ax4.set_xlabel('Freq. (Hz)')

    if opto_times is not None:
        ax5 = f.add_subplot(gs[-1,:2])
        psth_opto = neuron.get_psth(df=opto_times,event=0,colors=cmap,window=[-20,20],binsize=2)
        raster_sc = neuron.get_raster(df=opto_times,event=0,window=[-20,20],binsize=2,plot=False)
        plt.title(f'OptoTag')
        mean_sem = np.mean(psth_opto['data'][0]['sem'])
        mean_val = np.mean(psth_opto['data'][0]['mean'])
        ax5.set_ylabel('sp/s')
        yymax = ax5.get_ylim()[-1]
        if (mean_sem*5+mean_val)>yymax:
            yymax = mean_sem*5+mean_val
        ax5.set_ylim(0,yymax)
        ax5.get_legend().remove()
        ax5.axvspan(0,10,color='c',alpha=0.4)

        ax6 = ax5.twinx()
        bb = np.arange(raster_sc['window'][0],raster_sc['window'][1],raster_sc['binsize'])
        yy,xx = np.where(raster_sc['data'][0])


        ax6.plot(bb[xx],yy,'k.',alpha=0.5)
        ax6.set_ylabel('stim no.',color='c')
        ax6.set_ylim(0,opto_times.shape[0])
    plt.tight_layout(w_pad=-1)

    return(f)

    # psth_dia_on = neuron.get_psth(df=dia_df[dia_df.cat < dia_df.cat.max()], event='on_sec', conditions='cat',
    #                        colors=cmap)


@click.command()
@click.argument('ks2_dir')
def main(ks2_dir,p_save=None):
    run_tensor=False

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

    mouse_id = re.search('m\d\d\d\d-\d\d',ks2_dir).group()
    gate_id = int(re.search('_g\d_',ks2_dir).group()[2])
    probe_id = int(re.search('imec\d',ks2_dir).group()[-1])
    prefix = f'{mouse_id}_g{gate_id}_imec{probe_id}'
    print(f'run identifier is: {prefix}')

    plt.style.use('seaborn-paper')
    epochs,breaths,aux = data.load_aux(ks2_dir)
    breaths = breaths.reset_index().drop('Var1',axis=1)
    spikes = data.load_filtered_spikes(ks2_dir)[0]

    phi = proc.calc_phase(aux['pleth'])
    dum = breaths['inhale_onsets'].dropna()*aux['sr']
    dum = dum.values.astype('int')
    phi = proc.shift_phi(phi,dum)
    opto_fn = glob.glob(os.path.join(ks2_dir,'../../*XD_8_0_10.adj.txt'))[0]
    try:
        opto_time = pd.read_csv(opto_fn,header=None)
    except:
        opto_time=None




    # Plot raster summaries
    plot_long_raster(spikes,breaths,epochs)
    plt.savefig(os.path.join(p_save,f'{prefix}_example_raster_long.png'),dpi=150)
    plt.close('all')
    viz.plot_raster_example(spikes,aux['sr'],aux['pleth'],aux['dia'],500,5)
    plt.savefig(os.path.join(p_save,f'{prefix}_example_raster_short.png'),dpi=150)
    plt.close('all')

    # Plot single cell summaries
    df_sc = pd.DataFrame()
    THETA = []
    L_DIR = []
    CELL_ID = []
    MOD_DEPTH = []
    epochs_time = np.arange(0,1501,300)
    df_phase_tune = pd.DataFrame()

    max_time = 1501
    t0 = 0
    tf = np.where(aux['t'] < max_time)[0][-1]
    if len(phi) > tf:
        phi_slice = phi[t0:tf]
    else:
        phi_slice = phi
        max_time = aux['t'][-1]

    for cell_id in tqdm(spikes['cell_id'].unique()):
        # print(cell_id)
        # f = viz.plot_cell_summary_over_time(spikes,cell_id,epochs_time,breaths,phi,aux['sr'])
        # f.savefig(os.path.join(p_save,f'summary_cellid{cell_id}.png'),dpi=150)
        plt.close('all')

        dum = pd.DataFrame()

        btrain = np.zeros_like(phi_slice)
        spt = spikes[spikes.cell_id==cell_id]['ts']
        spt = spt[spt<max_time-1]
        sp_idx = np.round(spt * aux['sr']).astype('int')
        sp_idx = sp_idx[sp_idx<len(btrain)]
        btrain[sp_idx] = 1
        if sum(btrain)<1000:
            continue
        rr,theta_k,theta,L_dir = proc.angular_response_hist(phi_slice,btrain,nbins=25)
        mod_depth = (np.max(rr)-np.min(rr))/np.mean(rr)

        dum[cell_id] = rr
        df_phase_tune = pd.concat([df_phase_tune,dum],axis=1)


        THETA.append(theta)
        L_DIR.append(L_dir)
        MOD_DEPTH.append(mod_depth)
        CELL_ID.append(cell_id)
    df_phase_tune.index= theta_k[:-1]
    df_sc['theta'] = THETA
    df_sc['l_dir'] = L_DIR
    df_sc['cell_id'] = CELL_ID
    df_sc['mod_depth'] = MOD_DEPTH

    df_sc.to_csv(os.path.join(p_save,f'{prefix}_phase_tuning_stats.csv'))
    df_phase_tune.to_csv(os.path.join(p_save,f'{prefix}_phase_tuning_curves.csv'))

    ordered_mod_depth = df_sc.sort_values('mod_depth',ascending=False)['cell_id'].values

    sc_dir = os.path.join(p_save,'sc_figs')
    try:
        os.makedirs(sc_dir)
    except:
        pass

    max_time = aux['t'][-1]
    for ii,cell_id in enumerate(ordered_mod_depth):
        print(ii)
        f = plot_single_cell_summary(spikes[spikes.ts<max_time],cell_id,epochs_time,breaths,phi,aux['sr'],opto_time)
        f.savefig(os.path.join(sc_dir,f'{prefix}_summary_modrank{ii}_cellid{cell_id}.png'),dpi=150)
        plt.close('all')



    # =====================
    # Run the tensor factorization without affine warping
    # =====================
    dd = spikes.groupby('cell_id').mean()['depth']
    raster,cell_id,bins = proc.bin_trains(spikes.ts,spikes.cell_id,max_time=epochs_time[-1],binsize=.005)
    burst_dur = breaths['duration_sec'].mean()
    pbi_dur = breaths['postBI'].mean()
    TT,raster_bins = models.raster2tensor(raster,bins,breaths['on_sec'],pre=0.1,post=burst_dur+pbi_dur)


    # Plot the mean population traces aligned to breath
    mean_breath = np.mean(TT,2)
    mean_breath = (mean_breath-np.mean(raster,1))/np.std(raster,1)
    mean_breath[np.isnan(mean_breath)] = 0

    order = np.argsort(np.nanargmax(mean_breath,0))

    f,ax = plt.subplots(nrows=1,ncols=2,figsize=(4,5))
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.sca(ax[0])
    scl = np.max(np.abs(mean_breath))
    plt.pcolormesh(raster_bins,np.arange(len(order)),mean_breath.T[order,:],cmap='RdBu_r',vmin=-scl,vmax=scl)
    plt.axvline(0,ls=':',color='k')
    plt.axvline(burst_dur,ls=':',color='g')
    plt.text(0,mean_breath.shape[1]-1,'Burst',ha='left',va='top',fontsize=8)
    plt.xlabel('time [s]')
    plt.ylabel('neurons')
    plt.title('Z-scored FR')
    # plt.colorbar()

    plt.sca(ax[1])
    scl = np.max(np.abs(mean_breath))
    breath_ordered = mean_breath[:,order]
    breath_ordered = breath_ordered[:,np.where(np.max(np.abs(breath_ordered),0)>0.1)[0]]
    cc = plt.pcolormesh(raster_bins,np.arange(breath_ordered.shape[1]),breath_ordered.T,cmap='RdBu_r',vmin=-scl,vmax=scl)
    plt.axvline(0,ls=':',color='k')
    plt.axvline(burst_dur,ls=':',color='g')
    plt.text(0,breath_ordered.shape[1]-1,'Burst',ha='left',va='top',fontsize=8)
    plt.xlabel('time [s]')
    plt.title('Z-scored >0.1')

    f.subplots_adjust(right=0.8)
    f.colorbar(cc, cax=cbar_ax)
    sns.despine()
    plt.savefig(os.path.join(p_save,f'{prefix}_mean_breath.png'),dpi=300)





    if run_tensor:
        # Fit Tensor decomp
        np.random.seed(42)
        mdl,axs = models.get_best_TCA(TT,max_rank=12,plot_tgl=True)
        plt.close('all')

        # plot tensor decomp
        factors = mdl.factors.factors
        viz.plot_tensortools_factors(mdl,raster_bins,dd)
        plt.savefig(os.path.join(p_save,'tensor_factors.png'),dpi=150)
        plt.close('all')

        with open(os.path.join(p_save,f'{prefix}tensor_decomp.pkl'),'wb') as fid:
            pickle.dump(factors,fid)

        # =====================
        # Try affine warping
        # =====================

        # This commented section is if you want to rebin the tensors.
        # raster,cell_id,bins = proc.bin_trains(spikes.ts,spikes.cell_id,max_time=epochs[-1],binsize=.025)
        # mask = np.sum(raster,1)>100
        # raster = raster[mask,:]
        # cell_id = cell_id[mask]
        # dd = dd.loc[np.where(mask)[0]].values
        # TT,raster_bins =models.raster2tensor(raster,bins,dia_df['on_sec'],pre=0.25,post=0.5)
        TTt = np.transpose(TT,[2,0,1])

        warp_mdl = PiecewiseWarping(n_knots=1)
        warp_mdl.fit(TTt,iterations=25)

        warped = warp_mdl.transform(TTt)
        np.save(os.path.join(p_save,f'{prefix}_warped_tensor.npy'),warped)
        warped = np.transpose(warped,[1,2,0])
        warp_decomp,axs = models.get_best_TCA(warped,max_rank=12,plot_tgl=True)
        f = viz.plot_tensortools_factors(warp_decomp,raster_bins,dd)

        # Save
        plt.savefig(os.path.join(p_save,f'{prefix}_tensor_factors_warped.png'),dpi=150)
        plt.close('all')

        with open(os.path.join(p_save,f'{prefix}_tensor_decomp_warped.pkl'),'wb') as fid:
            pickle.dump(factors,fid)


if __name__=='__main__':
    main()




















