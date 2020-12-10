"""Visualization routines."""
import seaborn as sns
from spykes.plot import NeuroVis
import numpy as np
import matplotlib.pyplot as plt
# from . import proc
import proc
import matplotlib.cm as cm
import pandas as pd

def spykes_raster(raster):
    tt = np.arange(raster['window'][0],raster['window'][1],raster['binsize'])
    plt.pcolormesh(tt,np.arange(raster['data'][0].shape[0]),raster['data'][0],cmap='Greys',vmax=1,vmin=0)
    plt.axvline()


def plot_cell_summary_over_time(spikes,neuron_id,epochs,dia_df,phi,sr):
    '''
    assumes continuous epochs
    :param spikes:
    :param neuron_id:
    :param epochs:
    :param dia_df:
    :param phi:
    :param sr:
    :return:
    '''
    ts = spikes[spikes.cell_id==neuron_id].ts.values
    depth = spikes[spikes.cell_id==neuron_id].depth.mean()

    f = plt.figure(figsize=(10,3))
    ax0 = f.add_subplot(131,projection='polar')

    epochs_t0 = epochs[:-1]
    epochs_tf = epochs[1:]
    cat = proc.events_in_epochs(dia_df['on_sec'], epochs)
    dia_df['cat'] = cat

    cmap = cm.copper(np.linspace(0,1,len(epochs)))
    for ii, (t0, tf) in enumerate(zip(epochs_t0, epochs_tf)):
        sub_spikes = ts[ts > t0]
        sub_spikes = sub_spikes[sub_spikes < tf]

        t0_samp = int(t0 * sr)
        tf_samp = int(tf * sr)
        phi_slice = phi[t0_samp:tf_samp]
        btrain = np.zeros_like(phi_slice)

        spt = sub_spikes - t0
        sp_idx = np.round(spt * sr).astype('int')
        btrain[sp_idx] = 1
        rr, theta_k, theta, L_dir = proc.angular_response_hist(phi_slice, btrain, 25)
        rr = np.hstack([rr,rr[0]])
        plt.polar(theta_k, rr * sr, color=cmap[ii])

    plt.xticks([0,np.pi/2,np.pi,3*np.pi/2])
    ax0.set_xticklabels(['Insp','mid-I','I-E','E'])

    plt.suptitle(f'Neuron {neuron_id}; loc:{depth:0.0f}um; mod_depth:{L_dir:0.2f}')
    neuron = NeuroVis(ts)
    ax1 = f.add_subplot(132)


    # This line is needed to throw out all time after last epoch, which is gnerally longer thean all specified epochs
    dia_df = dia_df[dia_df['cat']<dia_df.cat.max()]
    psth_dia_off = neuron.get_psth(df=dia_df, event='off_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=20)
    plt.title(f'Dia Off')
    legend_vals = ['Dia_off']
    legend_vals += [f'{x}s - {y}s' for x, y in zip(epochs_t0, epochs_tf)]
    leg = plt.legend([])
    ax1.get_legend().remove()

    ax2 = f.add_subplot(133,sharex=ax1,sharey=ax1)
    psth_dia_off = neuron.get_psth(df=dia_df, event='on_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=20)
    plt.title(f'Dia On')
    plt.ylabel('')

    legend_vals = ['Trigger']
    legend_vals += [f'{x}s - {y}s' for x, y in zip(epochs_t0, epochs_tf)]
    plt.legend(legend_vals,bbox_to_anchor=(1.6,0.5),loc='center right')
    plt.tight_layout()

    # psth_dia_on = neuron.get_psth(df=dia_df[dia_df.cat < dia_df.cat.max()], event='on_sec', conditions='cat',
    #                        colors=cmap)


def plot_cell_summary_disjointed(spikes,neuron_id,epochs,dia_df,phi,sr,epoch_labels=None):
    '''
    disjointed epochs
    :param spikes:
    :param neuron_id:
    :param epochs: an n x 2 array
    :param dia_df:
    :param phi:
    :param sr:
    :return:
    '''
    #TODO: Plot rasters
    ts = spikes[spikes.cell_id==neuron_id].ts.values
    depth = spikes[spikes.cell_id==neuron_id].depth.mean()

    f = plt.figure(figsize=(10,3))
    ax0 = f.add_subplot(131,projection='polar')

    epochs_t0 = epochs[:,0]
    epochs_tf = epochs[:,1]

    dia_df['cat'] = -1

    cmap = cm.Dark2(np.linspace(0,1,epochs.shape[0]))
    for ii, (t0, tf) in enumerate(zip(epochs_t0, epochs_tf)):
        sub_spikes = ts[ts > t0]
        sub_spikes = sub_spikes[sub_spikes < tf]
        mask = np.logical_and(
            dia_df['on_sec']<tf,
            dia_df['on_sec'] > t0
        )
        dia_df['cat'][mask] = ii

        t0_samp = int(t0 * sr)
        tf_samp = int(tf * sr)
        phi_slice = phi[t0_samp:tf_samp]
        btrain = np.zeros_like(phi_slice)

        spt = sub_spikes - t0
        sp_idx = np.round(spt * sr).astype('int')
        btrain[sp_idx] = 1
        rr, theta_k, theta, L_dir = proc.angular_response_hist(phi_slice, btrain, 25)
        rr = np.hstack([rr,rr[0]])
        plt.polar(theta_k, rr * sr, color=cmap[ii])

    plt.xticks([0,np.pi/2,np.pi,3*np.pi/2])
    ax0.set_xticklabels(['Insp','mid-I','I-E','E'])

    plt.suptitle(f'Neuron {neuron_id}; loc:{depth:0.0f}um; mod_depth:{L_dir:0.2f}')
    neuron = NeuroVis(ts)
    ax1 = f.add_subplot(132)


    # This line is needed to throw out all time after last epoch, which is gnerally longer thean all specified epochs
    dia_df = dia_df[dia_df['cat']>=0]
    psth_dia_off = neuron.get_psth(df=dia_df, event='off_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=20)
    plt.title(f'Dia Off')
    ax1.get_legend().remove()

    ax2 = f.add_subplot(133,sharex=ax1,sharey=ax1)
    psth_dia_off = neuron.get_psth(df=dia_df, event='on_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=20)
    plt.title(f'Dia On')
    plt.ylabel('')

    legend_vals = ['Trigger']
    if epoch_labels is None:
        legend_vals += [f'{x}s - {y}s' for x, y in zip(epochs_t0, epochs_tf)]
    else:
        legend_vals += epoch_labels
    plt.legend(legend_vals,bbox_to_anchor=(1.6,0.5),loc='center right')
    plt.tight_layout()

    # psth_dia_on = neuron.get_psth(df=dia_df[dia_df.cat < dia_df.cat.max()], event='on_sec', conditions='cat',
    #                        colors=cmap)

def plot_depth_map(vals,depths):
    '''
    Given a vector of values and depths plot a weighted histogram
    and the individual points
    :param vals:
    :param depths:
    :return:
    '''
    pass


def plot_raster_summary(spikes,phys_df):
    raster, cell_id, bins = proc.bin_trains(spikes.ts, spikes.cell_id, binsize=5, start_time=0, max_time=phys_df.index[-1])
    f = plt.figure(figsize=(4,7))
    gs = f.add_gridspec(7,1)
    ax1 = f.add_subplot(gs[:-2,0])
    ax1.pcolormesh(bins,cell_id,raster,cmap='Greys',vmin=0,vmax=np.percentile(raster,85))
    ax2 = f.add_subplot(gs[-2:-1,0],sharex=ax1)
    ax2.plot(phys_df.index,phys_df['heart_rate'],'.',alpha=0.6,ms=2,color= 'tab:green')

    ax3 = f.add_subplot(gs[-1:,0],sharex=ax1)
    ax3.plot(phys_df.index,phys_df['dia_rate'],'.',alpha=0.6,ms=2,color='tab:purple')


    ax1.axis('off')
    ax1.text(-phys_df.index[-1]/100,spikes.cell_id.nunique()/2,'Neurons',rotation=90,horizontalalignment='right')
    sns.despine()
    ax2.set_ylabel('Pulse (Hz)')
    ax3.set_ylabel('Dia Rate (Hz)')
    ax3.set_xlabel('Time (s)')
    aa = ax1.get_xlim()[-1]
    ax1.set_xlim([0,aa])
    plt.tight_layout()

    return(f,[ax1,ax2,ax3])


def plot_raster_example(spikes,sr,pleth,dia_int,t0,win=10,events=None):

    sub_spikes = spikes[spikes.ts>t0]
    sub_spikes = sub_spikes[sub_spikes.ts<(t0+win)]

    # Create a time vector for the analog data
    sub_samps = np.arange(t0*sr,(t0+win)*sr).astype('int')
    sub_tvec = np.arange(t0,t0+win,1/sr)

    # Create scale factors
    ymax = sub_spikes.depth.max()
    dia_sub = dia_int[sub_samps]
    dia_sub = dia_sub/np.max(dia_sub)*(0.1*ymax)
    pleth_sub = pleth[sub_samps]
    pleth_sub = pleth_sub/np.max(pleth_sub)*(0.1*ymax)



    f = plt.figure(figsize=(4,5))
    plt.plot(sub_spikes.ts,sub_spikes.depth,'k.',alpha=0.3,mew=0,ms=3)
    # plt.vlines(sub_spikes.ts,sub_spikes.depth,sub_spikes.depth+20,'k',alpha=1,lw=0.5)
    # plt.scatter(sub_spikes.ts,sub_spikes.depth,c=sub_spikes.depth,s=2,alpha=0.7,cmap='jet')
    plt.plot(sub_tvec,dia_sub-(0.1*ymax),lw=0.5,color='tab:green')
    plt.plot(sub_tvec,pleth_sub-(0.2*ymax),lw=0.5,color='tab:purple')
    plt.axis('off')

    ymin = np.min(pleth_sub)-(0.2*ymax)
    plt.hlines(ymin,t0,(t0+win/25))
    plt.text(t0,ymin-(0.01*ymax),f'{win/10}s',fontsize=8,verticalalignment='top')
    plt.vlines(t0-(win/25),0,500)
    plt.text(t0-(win/25),0,'500 $\\mu$m',rotation=90,horizontalalignment='right',verticalalignment='bottom',fontsize=8)

    plt.arrow(t0-(win/25),np.median(sub_spikes.depth),0,-500,width=0.1,head_length=250,color='k')
    plt.text(t0-(win/15),np.median(sub_spikes.depth),'Rostral',
             fontsize=8,
             horizontalalignment='right',verticalalignment='top',
             rotation=90)

    if events is not None:
        if type(events) is pd.core.series.Series:
            events = events.values
        events = events[events<(t0+win)]
        events = events[events>t0]
        # [y1,y2] = plt.gca().get_ylim()

        plt.vlines(events,ymin,ymax,ls='--',color='tab:red',alpha=0.4)

    plt.tight_layout()

    return(f)




