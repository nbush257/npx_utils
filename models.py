"""Routines for fitting models."""
import numpy as np
import tensortools as tt
import matplotlib.pyplot as plt
import pandas as pd
try:
    from npx_utils.proc import raster2tensor
except:
    from proc import raster2tensor
def get_best_TCA(TT,max_rank=15,plot_tgl=True):
    ''' Fit ensembles of tensor decompositions.
        Returns the best model with the fewest parameters
        '''
    methods = (
        'ncp_hals',  # fits nonnegative tensor decomposition.
    )
    ax=None
    axx=None
    if max_rank<3:
        max_rank=3
    ranks = range(1,max_rank+1)
    ensembles = {}
    m = methods[0]
    ensembles[m] = tt.Ensemble(fit_method=m, fit_options=dict(tol=1e-4))
    ensembles[m].fit(TT, ranks=ranks, replicates=10)

    if plot_tgl:
        plot_opts1 = {
            'line_kw': {
                'color': 'black',
                'label': 'ncp_hals',
            },
            'scatter_kw': {
                'color': 'black',
                'alpha':0.2,
            },
        }
        plot_opts2 = {
            'line_kw': {
                'color': 'red',
                'label': 'ncp_hals',
            },
            'scatter_kw': {
                'color': 'red',
                'alpha':0.2,
            },
        }
        plt.close('all')
        fig = plt.figure
        ax = tt.plot_similarity(ensembles[m],**plot_opts1)
        axx = ax.twinx()
        tt.plot_objective(ensembles[m],ax=axx,**plot_opts2)
    mm = []
    for ii in range(1,max_rank+1):
        mm.append(np.median(ensembles[m].objectives(ii)))
    mm = 1-np.array(mm)
    best = np.argmax(np.diff(np.diff(mm)))+1
    optimal = ensembles[m].results[ranks[best]]
    dum_obj = 1
    for decomp in optimal:
        if decomp.obj<dum_obj:
            best_decomp = decomp
            dum_obj = decomp.obj

    if plot_tgl:
        axx.vlines(ranks[best],axx.get_ylim()[0],axx.get_ylim()[1],lw=3,ls='--')
    return(best_decomp,[ax,axx])



def run_affine():
    '''
    Run the Alex williams affine warp code
    :return:
    '''
    pass


def NMF_classifier():
    '''
    Run a naive Non-negative decomposition
    to classify insp/exp... type cells.
    :return:
    '''
