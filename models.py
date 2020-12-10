"""Routines for fitting models."""
import numpy as np
import tensortools as tt
def raster2tensor(raster,raster_bins,events,pre = .100,post = .200):
    '''
    Given the binned spikerates over time and a series of events,
    creates a tensor that is shape [n_bins_per_trial,n_cells,n_trials]

    :param raster: a [time x neurons] array of spike rates
    :param raster_bins: array of times in seconds for each bin ( first dim of raster)
    :param events: array of times in seconds at which each event started
    :param pre: float, amount of time prior to event onset to consider a trial(positive, seconds)
    :param post: float, amount of time after event onset to consider a trial (positive, seconds)
    :return:
        raster_T    - [time x cells x trials] tensor of spike rates for each trial (event)
        bins        - array of values in seconds that describes the first dimension of raster_T
    '''
    dt = np.round(np.mean(np.diff(raster_bins)),5)
    trial_length = int(np.round((pre+post)/dt))
    keep_events = events>(raster_bins[0]-pre)
    events = events[keep_events]
    keep_events = events<(raster_bins[-1]-post)
    events = events[keep_events]

    raster_T = np.empty([trial_length,raster.shape[0],len(events)])

    for ii,evt in enumerate(events):
        t0 = evt-pre
        t1 = evt+post
        bin_lims = np.searchsorted(raster_bins,[t0,t1])
        xx = raster[:,bin_lims[0]:bin_lims[0]+trial_length].T
        raster_T[:,:,ii]= xx
    bins = np.arange(-pre,post,dt)
    return(raster_T,bins)

def get_best_TCA(TT,max_rank=15,plot_tgl=True):
    ''' Fit ensembles of tensor decompositions.
        Returns the best model with the fewest parameters
        '''
    methods = (
        'ncp_hals',  # fits nonnegative tensor decomposition.
    )
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


def classify_breaths(dia_df, dia_int, pleth):
    '''
    Classify periods of apnea, sigh, gasp.
    :return:
    '''
    pass


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
