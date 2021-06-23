import sys
sys.path.append('../')
import data
import ccg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.matlab as sio
import glob
import os
import pandas as pd
import click
import seaborn as sns
import datetime

plt.style.use('seaborn-talk')

def load_all_probes(gate_dir,debug=False):
    ks_dirs = glob.glob(os.path.join(gate_dir,'*imec*/*ks2'))
    spikes = pd.DataFrame()
    n_cells = 0
    for ks_dir in ks_dirs:
        meta = data.parse_dir(ks_dir)
        temp = data.load_filtered_spikes(ks_dir)[0]
        if debug:
            temp = temp.query('cell_id<100')
        temp['probe'] = int(meta['probe'][-1])
        temp = temp.eval('unique_id = cell_id +@n_cells')
        n_cells += temp['cell_id'].nunique()
        spikes = pd.concat([spikes,temp])
    return(spikes,ks_dirs)


def plot_connectivity_summary(ccg_adj,p_save,prefix,spikes,thresh=7):
    bds = spikes.groupby('probe').min()['unique_id'].values

    ec,ic = ccg.get_ccg_peaks(ccg_adj,thresh=thresh,shape='mat')
    graph = ccg.to_graph(ec,ic)
    delay_graph = ccg.to_delay_graph(graph,ccg_adj)
    yy =delay_graph.shape[0]

    f = plt.figure(figsize=(6,5))
    plt.pcolormesh(delay_graph,cmap='RdBu_r',vmin=-20,vmax=20)
    cax = plt.colorbar()
    plt.xlabel('cells')
    plt.ylabel('cells')
    cax.set_label('Delay (ms)')
    plt.ylim(0,yy)
    plt.xlim(0,yy)
    plt.axis('square')
    plt.tight_layout()
    plt.grid('on')
    plt.plot(np.arange(0,yy),np.arange(0,yy),'k',lw=1,ls='--')
    for bd in bds[1:]:
        plt.axvline(bd)
        plt.axhline(bd)
    plt.savefig(os.path.join(p_save,f'{prefix}_delay_graph.png'),dpi=300)


    c1,c2 = np.where(graph)
    good_ccgs,tvec = ccg.extract_ccg(graph,ccg_adj)
    f,ax = plt.subplots(nrows=10,sharex=True,figsize=(4,8),sharey=True)
    for ii in range(10):
        if ii>=len(good_ccgs):
            continue
        cc1 = c1[ii]
        cc2= c2[ii]
        ax[ii].plot(tvec,good_ccgs[ii],'k',lw=1)
        ax[ii].set_yticks([])
        ax[ii].axvline(0,color='tab:red',ls=':')
        ax[ii].set_title(f'{cc1} to {cc2}',fontsize='small')
    ax[0].set_xlim(-100,100)

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(p_save,f'{prefix}_example_ccgs.png'),dpi=300)


    data_out = {}
    data_out['graph'] = graph
    data_out['delay_graph'] = delay_graph
    data_out['good_ccgs'] = good_ccgs
    return(data_out)



@click.command()
@click.argument('gate_dir')
@click.option('--tf',default=1200)
@click.option('--thresh',default=7)
@click.option('--debug',is_flag=True)
def main(gate_dir,tf,thresh,debug):
    print('loading spikes...')
    spikes,ks_dirs = load_all_probes(gate_dir)
    print('loading auxiliary data...')
    epochs,breaths,aux_data = data.load_aux(ks_dirs[0])
    print('computing ccg...')

    # String ops
    dt = datetime.datetime.now()
    date_str = f'{dt.year}-{dt.month:02.0f}-{dt.day:02.0f}'
    p_save = f'/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/results/{date_str}_connectivity'
    meta = data.parse_dir(ks_dirs[0])
    prefix = f'{meta["mouse_id"]}_g{meta["gate"]}'

    # make save directory
    try:
        os.makedirs(p_save)
    except:
        pass

    if debug:
        print('Running debug version...')
        ss = spikes.query('unique_id<60')
        ccg_adj,ccg_raw,pre_synaptic,post_synaptic = ccg.compute_ccg(ss['ts'].values,ss['unique_id'].values,
                                                                     breaths['on_sec'].values,
                                                                     max_time=100,
                                                                     shape='mat',
                                                                     event_window=2
                                                                     )
    else:
        ccg_adj,ccg_raw,pre_synaptic,post_synaptic = ccg.compute_ccg(spikes['ts'].values,
                                                                     spikes['unique_id'].values,
                                                                     breaths['on_sec'].values,
                                                                     max_time=tf,
                                                                     event_window = 2,
                                                                     shape='mat')


    data_out = plot_connectivity_summary(ccg_adj,p_save,prefix,spikes,thresh=thresh)
    data_out['ccg_adj'] = ccg_adj
    data_out['raw_ccg'] = ccg_raw
    data_out['presynaptic'] = pre_synaptic
    data_out['postsynaptic'] = post_synaptic
    sio.savemat(os.path.join(p_save,f'{prefix}_connectivity_data.mat'),data_out)


if __name__=='__main__':
    main()


