import sys
sys.path.append('../')
import shutil
import data
import proc

from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gs
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import scipy.io.matlab as sio
from sklearn.preprocessing import StandardScaler
import datetime


def plot_pca(pca,X,bins,breaths,aux,epochs,opto_df,max_t):
    plt.style.use('seaborn-paper')

    f = plt.figure(figsize=(18,10))
    gs = f.add_gridspec(3,4)
    ax = f.add_subplot(gs[0,0])
    ax.plot(np.cumsum(pca.explained_variance_ratio_), 'ko--')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Num Components')
    ax.set_ylabel('Explained Variance')
    sns.despine()

    # ========================= #
    # 2D PCA #
    # ========================= #
    ax = f.add_subplot(gs[0,1])
    for ii in range(0,X.shape[0],500):
        start = ii
        stop = start+500
        ax.plot(X[start:stop+1,0],X[start:stop+1,1],c=plt.cm.viridis(ii/X.shape[0]),alpha=.2)

    cmap = plt.get_cmap('viridis',50)
    norm = mpl.colors.Normalize(vmin=0, vmax=bins[ii]/60)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=np.arange(0, bins[ii]/60,10))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    cbar.set_label('time (m)')
    sns.despine()

    # ====================== #
    # 3D PCA #
    # ====================== #
    cmap = plt.get_cmap('plasma',50)
    ax = f.add_subplot(gs[0,2:],projection='3d')
    ax.scatter3D(xs=X[:, 0], ys=X[:, 1], zs=X[:, 2], c=X[:, 3],cmap=plt.get_cmap('plasma'),
                 alpha=0.01, s=10)
    vmin = np.min(X[:,3])
    vmax = np.max(X[:,3])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=np.linspace(vmin,vmax,2))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    cbar.set_label('PC4')

    # ====================== #
    # Long time #
    # ====================== #

    ax = f.add_subplot(gs[1,:])
    ax.plot(bins, X[:, 0:4], lw=0.5)
    ax.legend(['PC1','PC2','PC3','PC4'])
    ax.set_ylabel('Latent Amp')
    ax = ax.twinx()
    ax.plot(breaths['on_sec'], 1 / breaths['postBI'], 'k.', alpha=0.5)
    ax.set_ylabel('Resp. Rate (Hz)')
    ax.set_ylim(0,12)
    sns.despine()
    # epochs = epochs.query('t0*60<@max_t')
    for k,v in epochs.iterrows():
        t_epoch = v['t0']* 60
        ax.axvline(t_epoch, c='k', ls='--', lw=2)
        ax.text(t_epoch,ax.get_ylim()[1],v['label'])

    for k,v in opto_df.iterrows():
        ax.axvspan(v['on_sec'],v['off_sec'],color='c',alpha=0.3)
    # ax.set_xlim(0, max_t)
    ax.set_xlabel('time (s)')

    # ====================== #
    # Short time #
    # ====================== #
    ax2 = f.add_subplot(gs[2,:])
    ax2.plot(bins, X[:, 0:3], lw=1)
    ax2.set_xlim(epochs.iloc[0,0]*60+100, epochs.iloc[0,0]*60+115)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Latent Amp')
    ax2.legend(['PC1', 'PC2', 'PC3'])
    ax2 = plt.gca().twinx()
    ax2.plot(aux['t'], aux['dia'] + 13,'k--')
    ax2.plot(aux['t'], aux['pleth'] + 13, color='tab:purple',ls='--')

    ax2.set_ylim(0, 19)
    ax2.set_yticks([])
    plt.legend(['dia','pleth'])
    sns.despine()
    plt.tight_layout()



def main(ks_dir,max_t):
    spikes,metrics = data.load_filtered_spikes(ks_dir)
    epochs, breaths, aux = data.load_aux(ks_dir)
    raster, cell_id, bins = proc.bin_trains(spikes['ts'], spikes['cell_id'], binsize=0.005)
    aa = gaussian_filter1d(raster, sigma=5, axis=1)
    aa[np.isnan(aa)] = 0
    bb = np.sqrt(aa).T
    bb[np.isnan(bb)] = 0
    bb[np.isinf(bb)] = 0
    last_time = np.searchsorted(bins, epochs.iloc[0, 1] * 60)
    pca = PCA(10)
    pca.fit(bb[:last_time,:])
    X = pca.transform(bb)

    ni_bin_fn = data.get_ni_bin_from_ks_dir(ks_dir)
    opto = data.get_ni_analog(ni_bin_fn,3)
    ni_sr = data.get_sr(ni_bin_fn)
    opto_df = data.get_opto_df(opto,0.5,ni_sr)
    plot_pca(pca,X,bins,breaths,aux,epochs,opto_df,max_t)

    data_mat = {}
    data_mat['X'] = X
    data_mat['t'] = bins
    return(data_mat)

if __name__ == '__main__':
    # Must be run on HPC
    data_fn = "/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/ks3_dirs_filtered_v2.csv"
    dt = datetime.datetime.now()
    date_str = f'{dt.year}-{dt.month:02.0f}-{dt.day:02.0f}'
    save_p = f'/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/results/{date_str}_PCA_trajectory_plots'
    if os.path.isdir(save_p):
        shutil.rmtree(save_p)
        os.makedirs(save_p)
    else:
        os.makedirs(save_p)
    data_list = pd.read_csv(data_fn,header=None)
    for k,v in data_list.iterrows():
        try:
            ks_dir = v[0]
            max_t = v[1]
            print('='*100)
            print(f'Working on {ks_dir}')
            print('='*100)
            data_mat = main(ks_dir,max_t)
            sess = data.parse_dir(ks_dir)
            save_name = f'{save_p}/{sess["mouse_id"]}_g{sess["gate"]}_{sess["probe"]}_pca_trajectories.png'
            plt.savefig(save_name,dpi=300)

            save_name = f'{save_p}/{sess["mouse_id"]}_g{sess["gate"]}_{sess["probe"]}_pca_decomp.mat'
            sio.savemat(save_name,data_mat)
        except:
            print('ERROR!')
            print('ERROR!')
            print('ERROR!')

