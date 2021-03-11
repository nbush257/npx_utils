import click
import datetime
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
from tqdm.contrib import tenumerate
import re
import elephant
import neo
import quantities as pq
import scipy.io.matlab as sio
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


def transform_phase(phi):
    '''
    transforms phase from [0,1] to [-pi,pi] such that dia onset is at phi=0,offset is phi=np.pi
    :param phi:
    :return: phi_transform
    '''

    aa = phi+0.5
    aa[aa>1] = aa[aa>1]-1
    aa -=.5
    aa*=np.pi*2
    return(aa)



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
    ax2.set_ylim(0,10)

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


def plot_single_cell_summary(spikes, neuron_id, epoch_t, dia_df, phi, sr, opto_times,aux,is_tagged,stim_len):
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
    binsize=10
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
    f = plt.figure(figsize=(9,8))
    gs = f.add_gridspec(nrows=5,ncols=6)
    ax0 = f.add_subplot(gs[0,:2],projection='polar')

    # Plot the polar rsponses
    cmap = cm.copper(np.linspace(0, 1, len(epoch_t)))
    KL = []


    sig = neo.AnalogSignal(aux['dia'], units='V', sampling_rate=aux['sr'] * pq.Hz)
    n_train = neo.SpikeTrain(ts[ts<epoch_t[-1]], t_stop=epoch_t[-1] * pq.s, units=pq.s)
    sfc, freqs = elephant.sta.spike_field_coherence(sig, n_train, nperseg=4096)
    COH = np.max(sfc.magnitude)

    for ii, (t0, tf) in enumerate(zip(epochs_t0, epochs_tf)):
        sub_spikes = ts[ts > t0]
        sub_spikes = sub_spikes[sub_spikes < tf]

        t0_samp = int(t0 * sr)
        tf_samp = int(tf * sr)
        phi_slice = phi[t0_samp:tf_samp]
        btrain = np.zeros_like(phi_slice)

        spt = sub_spikes - t0
        sp_idx = np.round(spt * sr).astype('int')-1
        sp_idx = sp_idx[sp_idx<btrain.shape[0]]
        btrain[sp_idx] = 1
        rr, theta_k, theta, L_dir = proc.angular_response_hist(phi_slice, btrain, 25)
        kl = proc.compute_KL(phi_slice,btrain,25)
        KL.append(kl)
        rr = np.hstack([rr,rr[0]])
        plt.polar(theta_k, rr * sr, color=cmap[ii])


    ax0.set_xticks([0,np.pi/2,np.pi,3*np.pi/2])
    ax0.set_xticklabels(['I_on','','I_off','pre-I'],fontsize=6)

    plt.suptitle(f'Neuron {neuron_id}; loc:{depth:0.0f}um; Coh.:{np.mean(COH):0.2f}')

    # =======================
    # Get diaphragm triggered
    ax1 = f.add_subplot(gs[1,:2])
    # This line is needed to throw out all time after last epoch, which is gnerally longer thean all specified epochs
    dia_df = dia_df[dia_df['cat']<dia_df.cat.max()]
    psth_dia_off = neuron.get_psth(df=dia_df, event='on_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=binsize)
    plt.title(f'Dia On')
    legend_vals = ['Dia On']
    leg = plt.legend([])
    ax1.get_legend().remove()

    # =======================
    # Diaphragm offset triggered
    ax2 = f.add_subplot(gs[3,:2],sharex=ax1,sharey=ax1)
    psth_dia_off = neuron.get_psth(df=dia_df, event='off_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=binsize)
    plt.title(f'Dia Off')
    plt.ylabel('')

    legend_vals = ['Trigger']
    legend_vals += [f'{x/60}-{y/60}m' for x, y in zip(epochs_t0, epochs_tf)]
    plt.legend(legend_vals,bbox_to_anchor=(1.2,0.5),loc='center right',fontsize=6)

    # =========================
    # long raster
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
    ax4.hlines(dia_df.query('type=="sigh"')['on_sec']/60,ax4.get_xlim()[0],ax4.get_xlim()[1],color=cm.Dark2([3]),linestyles='--',lw=0.5)
    ax4.set_ylim(ax3.get_ylim())
    ax4.set_xlim(0,10)
    ax4.set_xlabel('Freq. (Hz)')

    ax7 = f.add_subplot(gs[2,:2])
    psth_dia_off = neuron.get_psth(df=dia_df.query('type!="apnea"'), event='on_sec', colors= cm.Dark2([2,3]),conditions='type',window=[-500,1000],binsize=50)
    plt.title(f'Eup/sigh')
    legend_vals = ['Dia On','eupnea','sigh']
    leg = plt.legend(legend_vals)
    plt.legend(legend_vals,bbox_to_anchor=(1.2,0.5),loc='center right',fontsize=6)
    maxy = max([ax1.get_ylim()[1],ax2.get_ylim()[1],ax7.get_ylim()[1]])
    ax1.set_ylim([0,maxy])
    ax2.set_ylim([0,maxy])
    ax7.set_ylim([0,maxy])
    ax7.axvline(0,color='k',ls='--')

    if opto_times is not None:
        ax5 = f.add_subplot(gs[-1,:2])
        psth_opto = neuron.get_psth(df=opto_times,event=0,colors=cmap,window=[-stim_len*2,stim_len*2],binsize=2)
        raster_sc = neuron.get_raster(df=opto_times,event=0,window=[-stim_len*2,stim_len*2],binsize=2,plot=False)
        if is_tagged:
            plt.title(f'OptoTag **')
        else:
            plt.title('OptoTag')
        mean_sem = np.mean(psth_opto['data'][0]['sem'])
        mean_val = np.mean(psth_opto['data'][0]['mean'])
        ax5.set_ylabel('sp/s')
        yymax = ax5.get_ylim()[-1]
        if (mean_sem*5+mean_val)>yymax:
            yymax = mean_sem*5+mean_val
        ax5.set_ylim(0,yymax)
        ax5.get_legend().remove()
        ax5.axvspan(0,stim_len,color='c',alpha=0.4)

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


def compute_opto_tag(ts,opto_time,stim_len):
    '''
    Determine if a cell has been optotaggd
    :param spikes:
    :param neuron_id:
    :param opto_time:
    :return:
    '''
    tagged = False
    first_time = opto_time.iloc[0]-1
    last_time = opto_time.iloc[-1]+1
    df = pd.DataFrame()

    neuron = NeuroVis(ts)
    psth_pre = neuron.get_psth(df=opto_time,event=0,binsize=stim_len,window=[-stim_len,0],plot=False)
    psth_post = neuron.get_psth(df=opto_time,event=0,binsize=stim_len,window=[0,stim_len],plot=False)
    raster = neuron.get_raster(df=opto_time,event=0,binsize=1,window=[-stim_len*2,stim_len*2],plot=False)
    raster = raster['data'][0]
    post_spikes = neuron.get_spikecounts(event=0,df=opto_time,window=[1,stim_len])
    n_stims = opto_time.shape[0]
    if n_stims*0.10> np.sum(post_spikes):
        return(raster,tagged)
    pre = psth_pre['data'][0]
    post = psth_post['data'][0]
    if post['mean'] > (pre['mean']+3*pre['sem']):
        tagged=True
    return(raster,tagged)


def plot_mean_breaths(spikes,breaths,max_time,p_save,prefix,coherence,coh_thresh=0.2):
    def mean_tensor(TT,raster):
        mean_breath = np.mean(TT, 2)
        mean_breath = (mean_breath - np.mean(raster, 1)) / np.std(raster, 1)
        mean_breath[np.isnan(mean_breath)] = 0
        return(mean_breath)

    def do_plot(mean_dat,scl = None,good_neurons=None):
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 5))
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        plt.sca(ax[0])
        if scl is None:
            scl = np.max(np.abs(mean_dat))

        cc = plt.pcolormesh(raster_bins, np.arange(len(order)), mean_dat.T[order, :], cmap='RdBu_r', vmin=-scl, vmax=scl)
        plt.axvline(0, ls=':', color='k')
        plt.axvline(burst_dur, ls=':', color='g')
        plt.text(0, mean_dat.shape[1] - 1, 'Burst', ha='left', va='top', fontsize=8)
        plt.xlabel('time [s]')
        plt.ylabel('neurons')
        plt.title('Z-scored FR')
        # plt.colorbar()

        plt.sca(ax[1])
        breath_ordered = mean_dat[:, order]
        coherence_ordered = np.array(coherence)[order]
        good_neurons = coherence_ordered>coh_thresh
        breath_ordered = breath_ordered[:, good_neurons]
        if np.sum(good_neurons)>0:
            cc = plt.pcolormesh(raster_bins, np.arange(breath_ordered.shape[1]), breath_ordered.T, cmap='RdBu_r', vmin=-scl,
                                vmax=scl)
            plt.axvline(0, ls=':', color='k')
            plt.axvline(burst_dur, ls=':', color='g')
            plt.text(0, breath_ordered.shape[1] - 1, 'Burst', ha='left', va='top', fontsize=8)
            plt.xlabel('time [s]')
            plt.title(f'Coh > {coh_thresh:0.1f}')

        f.subplots_adjust(right=0.8)
        f.colorbar(cc, cax=cbar_ax)
        sns.despine()
        return(good_neurons)

    raster,cell_id,bins = proc.bin_trains(spikes.ts,spikes.cell_id,max_time=max_time,binsize=.015)
    burst_dur = breaths['duration_sec'].mean()
    pbi_dur = breaths['postBI'].mean()

    TT_eup,raster_bins = models.raster2tensor(raster,bins,breaths.query('type=="eupnea"')['on_sec'],pre=0.1,post=burst_dur+pbi_dur)
    # TT_ap,raster_bins = models.raster2tensor(raster,bins,breaths.query('type=="apnea"')['on_sec'],pre=0.1,post=burst_dur+pbi_dur)
    # TT_sigh,raster_bins = models.raster2tensor(raster,bins,breaths.query('type=="sigh"')['on_sec'],pre=0.1,post=burst_dur+pbi_dur)

    mean_eup = mean_tensor(TT_eup,raster)
    # mean_ap = mean_tensor(TT_ap,raster)
    # mean_sigh = mean_tensor(TT_sigh,raster)

    order = np.argsort(np.nanargmax(mean_eup,0))
    scl = np.max(np.abs(mean_eup))
    good_neurons = do_plot(mean_eup,scl)
    # do_plot(mean_ap,scl,good_neurons)
    # do_plot(mean_sigh,scl=None,good_neurons = good_neurons)
    # do_plot(mean_sigh-mean_eup,good_neurons=good_neurons)

    plt.savefig(os.path.join(p_save,f'{prefix}_mean_breath.png'),dpi=300)


def get_opto_data(ks2_dir,stim_len):
    '''
    sequentially tries to get the opto data
    :param ks2_dir:
    :param stim_len:
    :return:
    '''
    try:
        opto_fn = glob.glob(os.path.join(ks2_dir, f'../../*XD_8_0_{stim_len}.txt'))[0]
        opto_time = pd.read_csv(opto_fn,header=None)
    except:
        try:
            opto_fn = glob.glob(os.path.join(ks2_dir, '../../*XD_8_0_10.adj.txt'))[0]
            opto_time = pd.read_csv(opto_fn, header=None)
            stim_len = 10
        except:
            try:
                opto_fn = glob.glob(os.path.join(ks2_dir, '../../*XD_8_0_10.txt'))[0]
                opto_time = pd.read_csv(opto_fn, header=None)
                stim_len = 10
            except:
                opto_fn = None
                opto_time = None
    if opto_fn is not None:
        print(f'Opto tag data found in: {os.path.split(opto_fn)[-1]}')
    else:
        print('No optotag data found')
    return(opto_time,opto_fn,stim_len)


@click.command()
@click.argument('ks2_dir')
@click.option('--t_max',default=1501)
@click.option('--stim_len',default=10)
def main(ks2_dir,t_max,stim_len,p_save=None):
    run_tensor=True

    # (DONE) Plot long time-scale raster of all neurons with respiratory rate
    # (DONE) Plot short timescale raster of all neurons with pleth and dia integrated
    # (DONE) Plot respiratory rate and heart rate
    # (DONE) Plot phase tuning and event psth of dia on for each neuron individually.
    #   Include gasp tuning (?) Include apnea tuning (?)
    # (Done)Plot event triggered raster of several example breaths
    # (DONE) Plot raster/psth of opto-triggered respones.
    # (DONE) Organize plots into a single image
    # Plot depth map of phase tuning.
    # Plot depth map of Tuning depth
    # Plot depth map of tagged neurons
    # Plot Dia/pleth and onsets to verify the detections
    # (DONE) Compute tagged boolean for all neuron
    # Compute recording depth, preferred phase, modulation depth, and is-modulated for all neurons
    # (DONE) Save Unit level data in csv

    # NEEDS: ks2dir,ni_binary, epoch times + labels, channels

    # ========================= #
    #  LOAD DATA #
    # ========================= #
    dt = datetime.datetime.now()
    date_str = f'{dt.year}-{dt.month:02.0f}-{dt.day:02.0f}'
    if p_save is None:
        p_results = f'/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/results/{date_str}_standard_postproc'
        p_sc = f'/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/results/{date_str}_standard_postproc/sc_figs'
    else:
        p_results =p_save
        p_sc = p_save

    try:
        os.makedirs(p_results)
    except:
        pass
    try:
        os.makedirs(p_sc)
    except:
        pass


    mouse_id = re.search('m\d\d\d\d-\d\d',ks2_dir).group()
    gate_id = int(re.search('_g\d_',ks2_dir).group()[2])
    probe_id = int(re.search('imec\d',ks2_dir).group()[-1])
    prefix = f'{mouse_id}_g{gate_id}_imec{probe_id}'
    print('='*100)
    print(f'run identifier is: {prefix}')

    plt.style.use('seaborn-paper')
    epochs,breaths,aux = data.load_aux(ks2_dir)
    max_time = np.min([aux['t'][-1],t_max])
    print(f'Max time is: {max_time:0.1f}s or {max_time/60:0.0f}m')
    breaths = proc.compute_breath_type(breaths) # classify sighs, apneas
    spikes = data.load_filtered_spikes(ks2_dir)[0]
    phi = proc.calc_dia_phase(breaths['on_sec'].values,breaths['off_sec'].values,dt=1/aux['sr'],t_stop=max_time)[1]
    phi = transform_phase(phi)
    opto_time,opto_fn,stim_len = get_opto_data(ks2_dir,stim_len)




    # =============================== #
    # Plot raster summaries
    plot_long_raster(spikes,breaths,epochs)
    plt.savefig(os.path.join(p_results,f'{prefix}_example_raster_long.png'),dpi=150)
    plt.close('all')
    if max_time<105:
        raster_plot_start = max_time-25
    else:
        raster_plot_start = 100
    viz.plot_raster_example(spikes,aux['sr'],aux['pleth'],aux['dia'],raster_plot_start,5)
    plt.savefig(os.path.join(p_results,f'{prefix}_example_raster_short.png'),dpi=150)
    plt.close('all')



    # =============================== #
    # Calculate and plot single cell summaries
    df_sc = pd.DataFrame()
    THETA = []
    L_DIR = []
    CELL_ID = []
    MOD_DEPTH = []
    COH=[]
    LAG = []
    TAGGED = []
    TAG_RASTER = []
    epochs_time = np.arange(0,max_time+1,300)
    if len(epochs_time)==1:
        epochs_time = np.concatenate([epochs_time,[max_time]])

    df_phase_tune = pd.DataFrame()

    t0 = 0
    tf = np.where(aux['t'] < max_time)[0][-1]
    if len(phi) >= max_time*aux['sr']:
        phi_slice = phi[t0:tf]
    else:
        phi_slice = phi
        max_time = aux['t'][-1]

    # Run this first to rank order the coherence
    print('Computing single cell characteristics...')
    for cell_id in tqdm(spikes['cell_id'].unique()):

         # Get phase tuning
        dum = pd.DataFrame()
        btrain = np.zeros_like(phi_slice)
        all_spt = spikes[spikes.cell_id==cell_id]['ts']
        spt = all_spt[all_spt<max_time-1]
        sp_idx = np.round(spt * aux['sr']).astype('int')
        sp_idx = sp_idx[sp_idx<len(btrain)]
        btrain[sp_idx] = 1
        if sum(btrain)<1000:
            continue
        rr,theta_k,theta,L_dir = proc.angular_response_hist(phi_slice,btrain,nbins=25)
        mod_depth = (np.max(rr)-np.min(rr))/np.mean(rr)
        LAG.append(theta_k[:-1][np.argmax(rr)])
        if opto_time is not None:
            tag_raster,tagged = compute_opto_tag(all_spt,opto_time,stim_len)
        else:
            tagged = False
            tag_raster = []
        TAGGED.append(tagged)
        TAG_RASTER.append(tag_raster)

    # Get coherence to diaphragm
        n_train = neo.SpikeTrain(spt,t_stop=max_time*pq.s,units=pq.s)
        sig = neo.AnalogSignal(aux['dia'],units='V',sampling_rate=aux['sr']*pq.Hz)
        sfc,freqs = elephant.sta.spike_field_coherence(sig,n_train,nperseg=4096)
        COH.append(np.max(sfc.magnitude))

        # Get phase tuning over time
        nid = prefix+f'_c{cell_id:03.0f}'
        dum[nid] = rr
        df_phase_tune = pd.concat([df_phase_tune,dum],axis=1)

        THETA.append(theta)
        L_DIR.append(L_dir)
        MOD_DEPTH.append(mod_depth)
        CELL_ID.append(cell_id)

    TAG_RASTER = np.array(TAG_RASTER)

    # get the coherences in order to threshold the mean breath plot
    coh_idx = np.zeros(spikes['cell_id'].nunique())
    coh_idx[np.array(CELL_ID)] = np.array(COH)
    plot_mean_breaths(spikes,breaths,max_time,p_results,prefix,coh_idx,coh_thresh=0.15)

    df_phase_tune.index= theta_k[:-1]
    df_sc['theta'] = THETA
    df_sc['l_dir'] = L_DIR
    df_sc['cell_id'] = CELL_ID
    df_sc['mod_depth'] = MOD_DEPTH
    df_sc['coherence'] = COH
    df_sc['phase_lag'] = LAG
    df_sc['tagged'] = TAGGED

    dd = spikes.groupby(['cell_id']).mean()['depth'].reset_index()
    df_sc = df_sc.merge(dd,how='inner',on='cell_id')
    df_sc['mouse_id'] = mouse_id
    df_sc['gate_id'] = gate_id
    df_sc['probe_id'] = probe_id
    df_sc['uid'] = prefix
    df_sc.to_csv(os.path.join(p_results,f'{prefix}_phase_tuning_stats.csv'))
    df_phase_tune.to_csv(os.path.join(p_results,f'{prefix}_phase_tuning_curves.csv'))
    sio.savemat(os.path.join(p_results,f'{prefix}_tag_rasters.mat'),{'t':np.arange(-0.02,0.02,0.001),'raster':TAG_RASTER,'cell_id':CELL_ID})

    ordered_mod_depth = df_sc.sort_values('coherence',ascending=False)['cell_id'].values

    # Plot the single cell data
    print('Plotting single cell summaries...')
    for ii,cell_id in tenumerate(ordered_mod_depth):
        is_tagged = df_sc.query('cell_id==@cell_id')['tagged'].values
        f = plot_single_cell_summary(spikes[spikes.ts<aux['t'][-1]],cell_id,epochs_time,breaths,phi,aux['sr'],opto_time,aux,is_tagged,stim_len)
        if is_tagged:
            f.savefig(os.path.join(p_sc,f'{prefix}_summary_modrank{ii[0]:03.0f}_cellid{cell_id}_tagged.png'),dpi=150)
        else:
            f.savefig(os.path.join(p_sc,f'{prefix}_summary_modrank{ii[0]:03.0f}_cellid{cell_id}.png'),dpi=150)
        plt.close('all')


    # =====================
    # Run the tensor factorization without affine warping
    # =====================
    dd = spikes.groupby('cell_id').mean()['depth']
    raster,cell_id,bins = proc.bin_trains(spikes.ts,spikes.cell_id,max_time=epochs_time[-1],binsize=.005)
    burst_dur = breaths['duration_sec'].mean()
    pbi_dur = breaths['postBI'].mean()
    TT,raster_bins = models.raster2tensor(raster,bins,breaths['on_sec'],pre=0.1,post=burst_dur+pbi_dur)



    if run_tensor:
        # Fit Tensor decomp
        np.random.seed(42)
        mdl,axs = models.get_best_TCA(TT,max_rank=12,plot_tgl=True)
        plt.close('all')

        # plot tensor decomp
        factors = mdl.factors.factors
        try:
            viz.plot_tensortools_factors(mdl,raster_bins,dd)
            plt.savefig(os.path.join(p_results,f'{prefix}_tensor_factors.png'),dpi=150)
            plt.close('all')
        except:
            pass

        with open(os.path.join(p_results,f'{prefix}tensor_decomp.pkl'),'wb') as fid:
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
        np.save(os.path.join(p_results,f'{prefix}_warped_tensor.npy'),warped)
        warped = np.transpose(warped,[1,2,0])
        warp_decomp,axs = models.get_best_TCA(warped,max_rank=12,plot_tgl=True)
        try:
            f = viz.plot_tensortools_factors(warp_decomp,raster_bins,dd)

            # Save
            plt.savefig(os.path.join(p_results,f'{prefix}_tensor_factors_warped.png'),dpi=150)
            plt.close('all')
        except:
            pass

        with open(os.path.join(p_results,f'{prefix}_tensor_decomp_warped.pkl'),'wb') as fid:
            pickle.dump(factors,fid)
    print('Analysis succesful')


if __name__=='__main__':
    main()




















