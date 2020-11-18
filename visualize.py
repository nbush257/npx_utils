"""Visualization routines."""
from spykes.plot import NeuroVis
import numpy as np
import matplotlib.pyplot as plt
from . import proc
import matplotlib.cm as cm

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

