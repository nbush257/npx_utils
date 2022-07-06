'''
Based on the Xiaowen Jia code from the allen. Using a different munging technique where
instead of having PSTHs for the stims, we are using raw spike times, jittering them, and
calculating the ISI distribution, not the CCG. Might need to go back to the CCG, but that would be easy
'''
import numpy as np
import scipy.signal
import sys

import proc
from tqdm import tqdm
sys.path.append('/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/dynaresp')
sys.path.append('/active/ramirez_j/ramirezlab/nbush/projects')
from npx_utils import proc
import utils
import os



def test(ts,cell_ids,t0,tf):
    '''
    This takes some time, and requires a good bit of memory.
    Gets all the spiketimes, jitters with 10ms uniform random, computes
    the true cross ISI distribution, and computes the jittered ISI distribution
    :param ts:
    :param cell_ids:
    :param t0:
    :param tf:
    :return:
    '''
    hist_bins = np.arange(-.05,.05,.001)
    idx = np.logical_and(ts>t0,ts<tf)
    ts_slice = ts[idx]
    cid_slice = cell_ids[idx]
    n_cells = len(np.unique(cell_ids))

    eps = np.random.uniform(-0.01,0.01,size=len(ts_slice))
    ts_j = ts_slice + eps

    # Create a dict so we only compute the spiketrains once:
    s_dict = {x:ts_slice[cid_slice==x] for x in np.unique(cell_ids)}
    s_dict_j = {x:ts_j[cid_slice==x] for x in np.unique(cell_ids)}

    out_mat = np.zeros([n_cells,n_cells,len(hist_bins)-1])

    for ii in tqdm(range(n_cells)):
        x = s_dict[ii]
        x_j = s_dict_j[ii]
        for jj in range(n_cells):
            # Skip if comparing same cell or if already computed
            if jj<=ii:
                continue
            y = s_dict[jj]
            y_j = s_dict_j[jj]

            ISI = np.subtract.outer(x,y).ravel()
            ISI = ISI[np.abs(ISI)<.05]

            ISI_j = np.subtract.outer(x_j,y_j).ravel()
            ISI_j = ISI_j[np.abs(ISI_j)<.05]


            hist_true=  np.histogram(ISI,hist_bins)[0]
            hist_j=  np.histogram(ISI_j,hist_bins)[0]

            hist_d = hist_true-hist_j
            out_mat[ii,jj,:] = hist_d
    return(hist_bins,out_mat)


def test2(ts,cell_ids,t0,tf):
    idx = np.logical_and(ts>t0,ts<tf)
    ts_slice = ts[idx]
    cid_slice = cell_ids[idx]
    n_cells = len(np.unique(cell_ids))

    # Jitter
    eps = np.random.uniform(-0.01,0.01,size=len(ts_slice))
    ts_j = ts_slice + eps

    raster_true,cc1,bins = proc.bin_trains(ts_slice,cid_slice,max_time = tf,start_time=t0,binsize=0.001)
    raster_jit,cc1,bins = proc.bin_trains(ts_j,cid_slice,max_time = tf,start_time=t0,binsize=0.001)

    out_true = np.zeros([n_cells,n_cells,200])
    out_jit = np.zeros([n_cells,n_cells,200])
    for ii in tqdm(cc1):
        x = raster_true[ii,:]
        x_j = raster_jit[ii,:]
        FR_i  = np.mean(x)*1000
        for jj in cc1:
            if jj>=ii:
                continue
            y = raster_true[jj, :]
            y_j = raster_jit[jj, :]
            FR_j = np.mean(y) * 1000
            # x = (x - np.mean(x)) / (np.std(x) * len(x))
            # y = (y - np.mean(y)) / (np.std(y))
            CCG_true = scipy.signal.correlate(x,y,mode='full')
            # x_j = (x_j - np.mean(x_j)) / (np.std(x_j) * len(x_j))
            # y_j = (y_j - np.mean(y_j)) / (np.std(y_j))
            CCG_jit = scipy.signal.correlate(x_j,y_j,mode='full')
            lags = np.arange(-len(x) + 1, len(x))/1000 # in s

            ctr = int(len(lags)/2)
            lags = lags[ctr-100:ctr+100]
            CCG_true = CCG_true[ctr-100:ctr+100]/(np.sqrt(FR_i*FR_j))
            CCG_jit = CCG_jit[ctr-100:ctr+100]/(np.sqrt(FR_i*FR_j))
            out_true[ii,jj,:] = CCG_true
            out_jit[ii, jj, :] = CCG_jit
    return(lags,out_true,out_jit)


# ======================= Below here is the old Xioawen Jia code ====================== #

def get_ccg_peaks(corrected_ccg,thresh = 7,shape='mat'):
    '''
    Assumes ccg is in milliseconds
    :param corrected_ccg:
    :return:
    '''

    if shape == 'ravel':
        centerpt = int(np.ceil(corrected_ccg.shape[0]/2))-1
        corrected_ccg[centerpt,:] = 0
        chopped = corrected_ccg[centerpt-100:centerpt+100,:]

        # Nan the middle 100ms to compute shoulder std
        dum = chopped.copy()
        dum[50:150] = np.nan
        shoulder_std = np.nanstd(dum,axis=0)

        # Look for peaks within 25ms
        center_only = chopped[75:125,:]
        compare_mat = np.tile(thresh*shoulder_std,[50,1])
        exc_connx = np.where(np.any(np.greater(center_only,compare_mat),axis=0))[0]
        inh_connx = np.where(np.any(np.less(center_only,-compare_mat),axis=0))[0]

        rm = np.where(shoulder_std==0)

        mask = np.logical_not(np.isin(exc_connx,rm))
        exc_connx = exc_connx[mask]

        mask = np.logical_not(np.isin(inh_connx,rm))
        inh_connx = inh_connx[mask]
    elif shape == 'mat':
        centerpt = int(np.ceil(corrected_ccg.shape[0]/2))-1
        corrected_ccg[centerpt,:,:] = 0
        chopped = corrected_ccg[centerpt-100:centerpt+100,:,:]

        # Nan the middle 100ms to compute shoulder std
        dum = chopped.copy()
        dum[50:150,:,:] = np.nan
        shoulder_std = np.nanstd(dum,axis=0)

        # Look for peaks within 25ms
        center_only = chopped[75:125,:,:]
        compare_mat = np.tile(thresh*shoulder_std,[50,1,1])
        exc_connx = np.any(np.greater(center_only,compare_mat),axis=0)
        inh_connx = np.any(np.less(center_only,-compare_mat),axis=0)

        rm = shoulder_std==0

        mask = np.logical_not(rm)

        exc_connx = exc_connx*mask
        inh_connx = inh_connx*mask


    return(exc_connx,inh_connx)


def to_graph(exc_connx,inh_connx):

    exc_connx = exc_connx.astype('int')
    inh_connx = -1*inh_connx.astype('int')

    test_double_count = exc_connx-inh_connx
    if np.any(np.abs(test_double_count)>1):
        print('MAJOR PROBLEMS! Counted a connectino as both inhibitory and excitatory')


    graph = exc_connx+inh_connx
    return(graph)


def to_delay_graph(graph,ccg):
    delay_graph = np.zeros_like(graph)
    centerpt = int(np.ceil(ccg.shape[0] / 2)) - 1
    connect_graph = np.abs(graph)
    cc1,cc2 = np.where(connect_graph)
    for c1,c2 in zip(cc1,cc2):
        this_ccg = ccg[:,c1,c2]
        lag = np.argmax(this_ccg)-centerpt
        delay_graph[c1,c2] = lag
        delay_graph[c2,c1] = -lag
    return(delay_graph)


def extract_ccg(graph,ccg):
    connect_graph = np.abs(graph)
    cc1,cc2 = np.where(connect_graph)
    ccg_examples = []
    centerpt = int(np.ceil(ccg.shape[0] / 2)) - 1
    tvec = np.arange(ccg.shape[0]) - centerpt
    for c1,c2 in zip(cc1,cc2):
        this_ccg = ccg[:,c1,c2]
        ccg_examples.append(this_ccg)
    return(ccg_examples,tvec)


if __name__=='__main__':
    print("THIS IS JUST A TEST")
    p_save = '/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/results'


    t0 = 10
    tf = 250
    dat = utils.load_mono_gate('m2021-48','g0')
    prb = dat['imec0']
    spikes,metrics = utils.prb_to_spikes(prb)
    ts = spikes['ts'].values
    cell_ids = spikes['cell_id'].values

    CCG = test2(ts, cell_ids, t0, tf)

    np.save(os.path.join(p_save,'ccg.npy'),CCG)
