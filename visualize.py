"""Visualization routines."""
import seaborn as sns
from spykes.plot import NeuroVis
import numpy as np
import matplotlib.pyplot as plt
try:
    from npx_utils import proc
    from npx_utils import models
except:
    import proc
    import models
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

    # Sample data
    ts = spikes[spikes.cell_id==neuron_id].ts.values
    if ts.shape[0]==0:
        return(0)
    depth = spikes[spikes.cell_id==neuron_id].depth.mean()
    epochs_t0 = epochs[:-1]
    epochs_tf = epochs[1:]
    cat = proc.events_in_epochs(dia_df['on_sec'], epochs)
    dia_df['cat'] = cat
    neuron = NeuroVis(ts)

    # Start plotting
    f = plt.figure(figsize=(4,6))
    gs = f.add_gridspec(nrows=3,ncols=2)
    ax0 = f.add_subplot(gs[0,0],projection='polar')

    # Plot the polar rsponses
    cmap = cm.copper(np.linspace(0,1,len(epochs)))
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
    ax1 = f.add_subplot(gs[1,0])
    # This line is needed to throw out all time after last epoch, which is gnerally longer thean all specified epochs
    dia_df = dia_df[dia_df['cat']<dia_df.cat.max()]
    psth_dia_off = neuron.get_psth(df=dia_df, event='on_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=20)
    plt.title(f'Dia On')
    legend_vals = ['Dia On']
    legend_vals += [f'{x/60}-{y/60}m' for x, y in zip(epochs_t0, epochs_tf)]
    leg = plt.legend([])
    ax1.get_legend().remove()

    ax2 = f.add_subplot(gs[2,0],sharex=ax1,sharey=ax1)
    psth_dia_off = neuron.get_psth(df=dia_df, event='off_sec', conditions='cat',colors=cmap,window=[-250,500],binsize=20)
    plt.title(f'Dia Off')
    plt.ylabel('')

    legend_vals = ['Trigger']
    legend_vals += [f'{x}s - {y}s' for x, y in zip(epochs_t0, epochs_tf)]
    plt.legend(legend_vals,bbox_to_anchor=(1.2,0.5),loc='center right',fontsize=6)

    ax3 = f.add_subplot(gs[:,1])
    raster = neuron.get_raster(df=dia_df,event='on_sec',window=[-500,500],binsize=5,plot=False)
    trial,tspike = np.where(raster['data'][0])
    bins = np.arange(raster['window'][0],raster['window'][1],raster['binsize'])
    ax3.plot(bins[tspike],dia_df['on_sec'][trial]/60,'k.',ms=2,alpha=0.1)
    ax3.set_ylabel('time [min]')
    ax3.set_xlabel('time [ms]')
    ax3.axvline(0,color='r',ls=':')
    ax3.set_ylim(epochs[0]/60,epochs[-1]/60)
    for ii, (t0, tf) in enumerate(zip(epochs_t0, epochs_tf)):
        ax3.axhspan(t0/60,tf/60,color=cmap[ii],alpha=0.1)
        ax3.axhline(t0/60,color='k',ls=':')

    plt.tight_layout()



    return(f)

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
    raster, cell_id, bins = proc.bin_trains(spikes.ts, spikes.cell_id, binsize=5, start_time=0, max_time=spikes.ts.iloc[-1])
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


def plot_raster_example(spikes,sr,pleth,dia_int,t0,win=10,events=None,pointsize=4):
    '''
    Plots the spike raster for [t0,t0+win]. Include the pleth and diaphragm traces.
    :param spikes: (DataFrame) spikes dataframe (columns are ts, cell_id)
    :param sr: (float)sample rate of the auxiliarry channels (Hz)
    :param pleth: (1D array) Pleth trace
    :param dia_int: (1D array) integrated diaphragm trace
    :param t0: (float) window start in seconds
    :param win: (float) window length in seconds)
    :param events:  (optional - list/array). Plots vertical lines at the times in events (seconds)
    :return:
    '''


    # subsample the spikes dataframe to the window
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

    # Set up the figure
    f = plt.figure(figsize=(4,5))
    # Plot the spike times as dots
    plt.plot(sub_spikes.ts,sub_spikes.depth,'k.',alpha=0.3,mew=0,ms=pointsize)

    # Plot the aux data - does some scaling to include everything in the same plot
    plt.plot(sub_tvec,dia_sub-(0.1*ymax),lw=0.5,color='tab:green')
    plt.plot(sub_tvec,pleth_sub-(0.2*ymax),lw=0.5,color='tab:purple')
    plt.axis('off')

    # Plot the scale bars
    ymin = np.min(pleth_sub)-(0.2*ymax)
    plt.hlines(ymin,t0,(t0+win/5),color='k')
    plt.text(t0,ymin-(0.01*ymax),f'{win/5:0.1f}s',fontsize=8,verticalalignment='top')
    plt.vlines(t0-(win/25),0,500,color='k')
    plt.text(t0-(win/25),0,'500 $\\mu$m',rotation=90,horizontalalignment='right',verticalalignment='bottom',fontsize=8)

    # Label the pleth/dia
    plt.text(t0,(-.1*ymax),'$\int$Dia',fontsize=8,rotation=90,horizontalalignment='right')
    plt.text(t0,(-.22*ymax),'Pleth',fontsize=8,rotation=90,horizontalalignment='right')

    # Draw the "rostral" arrow
    plt.arrow(t0-(win/25),np.median(sub_spikes.depth),0,-500,width=0.1,head_length=250,color='k')
    plt.text(t0-(win/15),np.median(sub_spikes.depth),'Rostral',
             fontsize=8,
             horizontalalignment='right',verticalalignment='top',
             rotation=90)

    # Plot the events as vertical lines
    if events is not None:
        if type(events) is pd.core.series.Series:
            events = events.values
        events = events[events<(t0+win)]
        events = events[events>t0]
        # [y1,y2] = plt.gca().get_ylim()

        plt.vlines(events,ymin,ymax,ls='--',color='tab:red',alpha=0.4)

    plt.tight_layout()

    return(f)


def plot_tensortools_factors(mdl,raster_bins,dd):
    '''
    Convinience function for plotting the tensor tools
    decompositions a little prettier than Alex Williams defaults
    Includes depth estimation of cells
    :param mdl: tensor tools decomp
    :param raster_bins: time axis from get_tensor
    :param dd: a 1D array of recording locations for each row of the cell- axis
    :return:  figure
    '''
    factors = mdl.factors.factors
    n_rank = factors[0].shape[1]
    n_neurons = factors[1].shape[0]
    f,ax = plt.subplots(nrows=n_rank,ncols=3,sharex='col',figsize=(5,7))
    cmap = np.tile(cm.Dark2(range(7)),[2,1])
    for ii in range(n_rank):
        ax[ii,0].plot(raster_bins*1000,factors[0][:,ii],lw=1,color=cmap[ii])
        ax[ii,0].axvline(0,color='r',ls=':')
        ax[ii,1].bar(dd,factors[1][:,ii],color=cmap[ii],width=100)
        ax[ii,2].plot(factors[2][:,ii],'.',ms=3,color=cmap[ii],alpha=0.4)
    sns.despine()
    ax[-1,0].set_xlabel('time [ms]')
    ax[-1,1].set_xlabel('depth [$\\mu m$]')
    ax[-1,2].set_xlabel('Breath no.')
    ax[0,0].set_title('Firing Patterns')
    ax[0,1].set_title('Cell Assemblies')
    ax[0,2].set_title('Network States')
    plt.tight_layout()
    return(f)


def plot_average_breaths_by_type(breaths,aux):

    def plotter(eup, sigh, apnea, t, ax=None):
        cmap = plt.cm.Dark2(np.arange(3))
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(111)

        m = np.median(eup, 1)
        iqr = np.percentile(eup, [25, 75], 1)
        ax.plot(t, m,color=cmap[0])
        ax.fill_between(t, iqr[0], iqr[1], color=cmap[0],alpha=0.2)

        m = np.median(apnea, 1)
        iqr = np.percentile(apnea, [25, 75], 1)
        ax.plot(t, m,color=cmap[1])
        ax.fill_between(t, iqr[0], iqr[1], color=cmap[1],alpha=0.2)

        m = np.median(sigh, 1)
        iqr = np.percentile(sigh, [25, 75], 1)
        ax.plot(t, m,color=cmap[2])
        ax.fill_between(t, iqr[0], iqr[1], color=cmap[2],alpha=0.2)

        return (ax)
    ep,ap,sp,t = models.get_breaths(breaths,aux['sr'],aux['pleth'])
    ed,ad,sd,t = models.get_breaths(breaths,aux['sr'],aux['dia'])
    f,ax = plt.subplots(nrows=2,figsize=(3,2),sharex=True)
    plotter(ep,sp,ap,t,ax=ax[0])
    plotter(ed,sd,ad,t,ax=ax[1])
    sns.despine()
    ax[0].axvline(0,ls=':',color='k')
    ax[1].axvline(0,ls=':',color='k')
    ax[0].set_ylabel('pleth')
    ax[1].set_ylabel('$\int$ dia')
    ax[1].legend(['eupnea','apnea','sigh'],fontsize=6,bbox_to_anchor=[0.8,1])
    plt.tight_layout()


def plot_reset_curve(breaths,events,opto_color='#00b7ff',t0=0,tf=None,annotate=False):
    if tf is None:
        tf = breaths['on_sec'].max()
    mean_IBI = breaths.query('on_sec>@t0 & on_sec<@tf')['IBI'].mean()
    mean_dur = breaths.query('on_sec>@t0 & on_sec<@tf')['duration_sec'].mean()
    xmax = []
    ymax = []
    for on_sec in np.random.uniform(low=t0, high=tf, size=(100,)):
        last_breath = breaths.query('on_sec<@on_sec').iloc[-1]
        next_breath = breaths.query('on_sec>@on_sec').iloc[0]

        t_since_last_off = on_sec - last_breath['on_sec']
        t_to_next_on = next_breath['on_sec'] - on_sec

        ctrls, = plt.plot(t_since_last_off, t_to_next_on + t_since_last_off, 'ko', ms=3, alpha=0.5)


    for on_sec in events:
        last_breath = breaths.query('on_sec<@on_sec').iloc[-1]
        next_breath = breaths.query('on_sec>@on_sec').iloc[0]

        t_since_last_off = on_sec - last_breath['on_sec']
        t_to_next_on = next_breath['on_sec'] - on_sec

        stims, = plt.plot(t_since_last_off, t_to_next_on + t_since_last_off, 'o', color=opto_color, mec='k', mew=0)
        xmax.append(t_since_last_off)
        ymax.append(t_to_next_on+t_since_last_off)

    xmax = max(xmax)
    ymax = max(ymax)

    plt.axvline(mean_dur, color='k', ls='--', lw=0.5)
    plt.axhline(mean_IBI, color='k', ls='--', lw=0.5)

    plt.plot([0, mean_dur + mean_IBI], [0, mean_IBI + mean_dur], color='tab:red')

    plt.xlabel('Time since last breath onset (s)')
    plt.ylabel('Total time between breaths (s)')

    plt.xlim([0,xmax])
    plt.ylim([0, ymax])

    # Phase advance
    pts = np.array([[mean_dur, mean_dur], [mean_IBI, mean_IBI], [mean_dur, mean_IBI]])
    plt.fill_between(pts[:, 0], pts[:, 1], color='tab:green', alpha=0.3)

    # Phase delay
    pts = np.array([[mean_dur, mean_IBI], [mean_IBI, mean_IBI], [mean_IBI + mean_dur, mean_IBI + mean_dur],
                    [mean_IBI + mean_dur, plt.gca().get_ylim()[1]], [mean_dur, plt.gca().get_ylim()[1]]])
    plt.fill(pts[:, 0], pts[:, 1], color='tab:grey', alpha=0.3)

    # Shorten inspiration
    pts = np.array([[0, 0], [mean_dur, mean_dur], [mean_dur, mean_IBI], [0, mean_IBI]])
    plt.fill(pts[:, 0], pts[:, 1], color='tab:purple', alpha=0.3)

    # Prolong inspiration
    pts = np.array(
        [[0, mean_IBI], [mean_dur, mean_IBI], [mean_dur, plt.gca().get_ylim()[1]], [0, plt.gca().get_ylim()[1]]])
    plt.fill(pts[:, 0], pts[:, 1], color='tab:orange', alpha=0.3)

    if annotate:
        plt.text(mean_dur, mean_dur / 2, 'inspiration\nduration', rotation=90)
        plt.text(0, (mean_dur + mean_IBI) / 2, 'Shorten inspiration', ha='left')
        plt.text(0, plt.gca().get_ylim()[1], 'Prolong inspiration', ha='left', va='top')
        plt.text(mean_dur, (mean_dur + mean_IBI) / 2, 'Advance phase', ha='left', va='center')
        plt.text(mean_dur, plt.gca().get_ylim()[1], 'Delay phase', ha='left', va='top')
        plt.text((mean_dur + mean_IBI) / 2, (mean_dur + mean_IBI) / 2, 'Lower bound', color='tab:red', ha='left', va='top')
        plt.legend([stims, ctrls], ['Stims', 'Random'])

    sns.despine()


def plot_reset_curve_normed(breaths,events,opto_color='#00b7ff',t0=0,tf=None,annotate=False,plot_tgl=True):
    '''
    Plots a reset curve in normalized time.
    :param breaths:
    :param events:
    :param opto_color:
    :param t0:
    :param tf:
    :param annotate:
    :return: x_stim,y_stim,x_control,y_control
    '''
    if tf is None:
        tf = breaths['on_sec'].max()
    mean_IBI = breaths.query('on_sec>@t0 & on_sec<@tf')['IBI'].mean()
    mean_dur = breaths.query('on_sec>@t0 & on_sec<@tf')['duration_sec'].mean()
    x_stim = []
    y_stim = []
    x_control=[]
    y_control= []
    for on_sec in np.random.uniform(low=100, high=600, size=(100,)):
        last_breath = breaths.query('on_sec<@on_sec').iloc[-1]
        next_breath = breaths.query('on_sec>@on_sec').iloc[0]

        t_since_last_on = on_sec - last_breath['on_sec']
        cycle_duration = next_breath['on_sec'] - last_breath['on_sec']

        if plot_tgl:
            ctrls, = plt.plot(t_since_last_on / mean_IBI, cycle_duration / mean_IBI, 'ko', ms=3, alpha=0.5)
        x_control.append(t_since_last_on/mean_IBI)
        y_control.append(cycle_duration/mean_IBI)

    for on_sec in events:
        last_breath = breaths.query('on_sec<@on_sec').iloc[-1]
        next_breath = breaths.query('on_sec>@on_sec').iloc[0]

        t_since_last_on = on_sec - last_breath['on_sec']
        cycle_duration = next_breath['on_sec'] - last_breath['on_sec']

        if plot_tgl:
            stims, = plt.plot(t_since_last_on / mean_IBI, cycle_duration / mean_IBI, 'o', color=opto_color, mec='k', mew=0)
        x_stim.append(t_since_last_on/mean_IBI)
        y_stim.append(cycle_duration/mean_IBI)

    # Remap output data to arrays
    x_stim = np.array(x_stim)
    y_stim = np.array(y_stim)
    x_control = np.array(x_control)
    y_control = np.array(y_control)

    # Skip plotting and just output data
    if not plot_tgl:
        return(x_stim,y_stim,x_control,y_control)


    # Essential plot features
    plt.axvline(mean_dur / mean_IBI, color='k', ls='--', lw=0.5)
    plt.axhline(1, color='k', ls='--', lw=0.5)
    plt.plot([0, 2], [0, 2], color='tab:red')
    plt.xlabel('Stim time (normalized)')
    plt.ylabel('Cycle duration (normalized)')
    plt.xlim(0, 1.5)
    plt.ylim(0, 2)
    plt.yticks([0, 1, 2])
    plt.xticks([0, 0.5, 1])

    # Accessory plot features
    if plot_tgl & annotate:
        plt.text(0.01, 1.5, 'Prolong inspiration', ha='left', va='bottom', rotation=90)
        plt.text(0.01, 0.01, 'Shorten inspiration', ha='left', va='bottom', rotation=90)
        plt.text(mean_dur / mean_IBI + 0.01, mean_dur / mean_IBI + 0.05, 'Phase advance', ha='left', va='bottom',
                 rotation=90)
        plt.text(mean_dur / mean_IBI + 0.01, 1.5, 'Phase delay', ha='left', va='bottom', rotation=90)

        plt.fill_between([0, mean_dur / mean_IBI], [0, mean_dur / mean_IBI], [1, 1], color='tab:purple', alpha=0.2)
        plt.fill_between([0, mean_dur / mean_IBI], [1, 1], [2, 2], color='tab:green', alpha=0.2)
        pts = np.array([[mean_dur / mean_IBI,1],[1,1],[1.5,1.5],[1.5,2],[mean_dur/mean_IBI,2]])
        plt.fill(pts[:,0],pts[:,1],color='tab:orange',alpha=0.2)
        # plt.fill_between([mean_dur / mean_IBI, 1], [1, 1], [2, 2], color='tab:orange', alpha=0.2)
        plt.fill_between([mean_dur / mean_IBI, 1], [mean_dur / mean_IBI, 1], [1, 1], color='tab:grey', alpha=0.2)

        plt.text(mean_dur / mean_IBI / 2, mean_dur / mean_IBI / 2 * 0.8, 'Lower bound', color='tab:red', rotation=26)

        plt.text(mean_dur / mean_IBI / 2, plt.gca().get_ylim()[1], 'Inspiration', ha='center', va='top')
        plt.text(mean_dur / mean_IBI + (1 - mean_dur / mean_IBI) / 2, plt.gca().get_ylim()[1], 'Expiration', ha='center',
                 va='top')

        plt.xlim(0, 1.5)
        plt.ylim(0, 2)
        plt.yticks([0, 1, 2])
        plt.xticks([0, 0.5, 1,1.5])

    sns.despine()
    return(x_stim,y_stim,x_control,y_control)

def plot_raster_with_aux(spikes,aux_t,aux,t0,tf,binsize=0.05,figsize=(10,10),cmap='magma',vmin=0,vmax=50):
    '''
    Plot a raster (heatmap) with the auxiliary channel above.
    param spikes: a spikes data frame
    param aux_t : time vector that maps the auxiliary channel into time (units: s)
    param aux: an auxiliary channel to plot over the raster (e.g., diaphragm)
    param t0: first time (in seconds) to plot
    param tf: last time (in seconds) to plot
    param binsize: binsize in seconds. Default=0.05 (50ms)

    return: ax_aux, ax_rast
    '''
    # Compute raster
    raster,cell_id,bins = proc.bin_trains(spikes['ts'],spikes['cell_id'],start_time=t0, max_time=tf,binsize=binsize)
    raster= raster/binsize

    # Set up figure
    gs = plt.GridSpec(10,1)
    f = plt.figure(figsize=figsize)
    ax_aux = f.add_subplot(gs[:1,:])

    # Plot auxiliary channel
    s0,sf = np.searchsorted(aux_t,[t0,tf])
    ax_aux.plot(aux_t[s0:sf],aux[s0:sf],color='k',lw=0.5)

    # Plot raster as a mesh
    ax_rast = f.add_subplot(gs[1:,:],sharex=ax_aux)
    ax_rast.pcolormesh(bins,cell_id,raster,cmap=cmap,vmin=vmin,vmax=vmax)

    # If there are multiple probes, plot a horizontal line
    try:
        probe_cuts = spikes.groupby('probe')['cell_id'].max().values
        for cc in probe_cuts:
            ax_rast.axhline(cc,color='w',lw=1)
    except:
        pass

    # Set the xlim and clean up
    plt.xlim(t0,tf)
    ax_aux.axis('off')

    return(ax_aux,ax_rast)

def plot_spikes_with_aux(spikes,aux_t,aux,t0,tf,figsize=(4,6),aux_ylim=None):
    '''
    Plot a scatterplot of spikes with the auxiliary channel above.
    param spikes: a spikes data frame
    param aux_t : time vector that maps the auxiliary channel into time (units: s)
    param aux: an auxiliary channel to plot over the raster (e.g., diaphragm)
    param t0: first time (in seconds) to plot
    param tf: last time (in seconds) to plot
    param binsize: binsize in seconds. Default=0.05 (50ms)

    return: ax_aux, ax_spk
    '''

    gs = plt.GridSpec(8,1)
    f = plt.figure(figsize=figsize)
    ax_aux = f.add_subplot(gs[:1,:])
    ax_spk = f.add_subplot(gs[1:,:],sharex=ax_aux)

    temp = spikes.query('ts>@t0 & ts<@tf')
    plt.sca(ax_spk)
    if 'probe' in temp.columns:
        sns.scatterplot(data=temp,x='ts',y='cell_id',hue='probe',s=4,palette=['k','r'],hue_order=['imec0','imec1'],legend=False)
    else:
        sns.scatterplot(data=temp,x='ts',y='cell_id',s=4,legend=False,color='k')
    s0,sf = np.searchsorted(aux_t,[t0,tf])
    ax_aux.plot(aux_t[s0:sf],aux[s0:sf],'k',lw=1)
    plt.xlim(t0,tf)
    if aux_ylim is not None:
        ax_aux.set_ylim(aux_ylim)

    sns.despine()
    return(ax_aux, ax_spk)