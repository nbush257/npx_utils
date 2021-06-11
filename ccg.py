import numpy as np
import proc
from tqdm import tqdm


def jitter(data, l):
    """
    Jittering multidemntational logical data where
    0 means no spikes in that time bin and 1 indicates a spike in that time bin.
     Be sure to cite Xiaoxuan Jia https://github.com/jiaxx/jitter
    """
    if len(np.shape(data)) > 3:
        flag = 1
        sd = np.shape(data)
        data = np.reshape(data, (
            np.shape(data)[0], np.shape(data)[1], len(data.flatten()) / (np.shape(data)[0] * np.shape(data)[1])),
                          order='F')
    else:
        flag = 0

    psth = np.mean(data, axis=1)
    length = np.shape(data)[0]

    if np.mod(np.shape(data)[0], l):
        data[length:(length + np.mod(-np.shape(data)[0], l)), :, :] = 0
        psth[length:(length + np.mod(-np.shape(data)[0], l)), :] = 0

    if len(np.shape(psth))>1 and np.shape(psth)[1] > 1:
        dataj = np.squeeze(
            np.sum(np.reshape(data, [l, np.shape(data)[0] // l, np.shape(data)[1], np.shape(data)[2]], order='F'),
                   axis=0))
        psthj = np.squeeze(
            np.sum(np.reshape(psth, [l, np.shape(psth)[0] // l, np.shape(psth)[1]], order='F'), axis=0))
    else:
        dataj = np.sum(np.reshape(data, [l, np.shape(data)[0] // l, np.shape(data)[1]], order='F'),axis=0)
        psthj = np.sum(np.reshape(psth, [l, np.shape(psth)[0] // l], order='F'),axis=0)

    if np.shape(data)[0] == l:
        dataj = np.reshape(dataj, [1, np.shape(dataj)[0], np.shape(dataj)[1]], order='F')
        psthj = np.reshape(psthj, [1, np.shape(psthj[0])], order='F')

    if len(np.shape(psthj))>1:
        psthj = np.reshape(psthj, [np.shape(psthj)[0], 1, np.shape(psthj)[1]], order='F')
    else:
        psthj = np.reshape(psthj, [np.shape(psthj)[0], 1], order='F')

    psthj[psthj == 0] = 10e-10

    if len(np.shape(psthj))>2:
        corr = dataj / np.tile(psthj, [1, np.shape(dataj)[1], 1])
        corr = np.reshape(corr, [1, np.shape(corr)[0], np.shape(corr)[1], np.shape(corr)[2]], order='F')
        corr = np.tile(corr, [l, 1, 1, 1])
        corr = np.reshape(corr, [np.shape(corr)[0] * np.shape(corr)[1], np.shape(corr)[2], np.shape(corr)[3]],
                          order='F')
        psth = np.reshape(psth, [np.shape(psth)[0], 1, np.shape(psth)[1]], order='F')
        output = np.tile(psth, [1, np.shape(corr)[1], 1]) * corr

        output = output[:length, :, :]
    else:
        corr = dataj / np.tile(psthj, [1, np.shape(dataj)[1]])
        corr = np.reshape(corr, [1, np.shape(corr)[0], np.shape(corr)[1]], order='F')
        corr = np.tile(corr, [l, 1, 1])
        corr = np.reshape(corr, [np.shape(corr)[0] * np.shape(corr)[1], np.shape(corr)[2],],
                          order='F')
        psth = np.reshape(psth, [np.shape(psth)[0], 1], order='F')
        output = np.tile(psth, [1, np.shape(corr)[1]]) * corr

        output = output[:length, :]


    return output


def xcorrfft(a,b,NFFT):
    CCG = np.fft.fftshift(np.fft.ifft(np.multiply(np.fft.fft(a,NFFT), np.conj(np.fft.fft(b,NFFT)))))
    return CCG


def nextpow2(n):
    """get the next power of 2 that's greater than n"""
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


def get_ccgjitter(spikes, FR, jitterwindow=25):
    # spikes: neuron*ori*trial*time
    assert np.shape(spikes)[0]==len(FR)

    n_unit=np.shape(spikes)[0]
    n_t = np.shape(spikes)[3]
    # triangle function
    t = np.arange(-(n_t-1),(n_t-1))
    theta = n_t-np.abs(t)
    del t
    NFFT = int(nextpow2(2*n_t))
    target = np.array([int(i) for i in NFFT/2+np.arange((-n_t+2),n_t)])

    ccgjitter = []
    rawccg = []
    pair=0
    for i in np.arange(n_unit-1): # V1 cell
        for m in np.arange(i+1,n_unit):  # V2 cell
            if FR[i]>2 and FR[m]>2:
                temp1 = np.squeeze(spikes[i,:,:,:])
                temp2 = np.squeeze(spikes[m,:,:,:])
                FR1 = np.squeeze(np.mean(np.sum(temp1,axis=2), axis=1))
                FR2 = np.squeeze(np.mean(np.sum(temp2,axis=2), axis=1))
                tempccg = xcorrfft(temp1,temp2,NFFT)
                tempccg = np.squeeze(np.nanmean(tempccg[:,:,target],axis=1))

                temp1 = np.rollaxis(np.rollaxis(temp1,2,0), 2,1)
                temp2 = np.rollaxis(np.rollaxis(temp2,2,0), 2,1)
                ttemp1 = jitter(temp1,jitterwindow)
                ttemp2 = jitter(temp2,jitterwindow)
                tempjitter = xcorrfft(np.rollaxis(np.rollaxis(ttemp1,2,0), 2,1),np.rollaxis(np.rollaxis(ttemp2,2,0), 2,1),NFFT)
                tempjitter = np.squeeze(np.nanmean(tempjitter[:,:,target],axis=1))
                ccgjitter.append((tempccg - tempjitter).T/np.multiply(np.tile(np.sqrt(FR[i]*FR[m]), (len(target), 1)),
                                                                      np.tile(theta.T.reshape(len(theta),1),(1,len(FR1)))))

    ccgjitter = np.array(ccgjitter)
    return ccgjitter



def jitter_NEB(data,l):
    psth = np.mean(data, axis=1)
    length = np.shape(data)[0]

    if np.mod(np.shape(data)[0], l):
        data[length:(length + np.mod(-np.shape(data)[0], l)), :] = 0
        psth[length:(length + np.mod(-np.shape(data)[0], l))] = 0

    # dataj = np.squeeze(np.sum(np.reshape(data,l,np.shape(data)[0] // l,np.shape(data)[1],order='F'),axis=0))
    dataj = np.sum(np.reshape(data, [l, np.shape(data)[0] // l, np.shape(data)[1]], order='F'),axis=0)
    psthj = np.sum(np.reshape(psth,[l,np.shape(psth)[0]//l],order='F'))

    corr = dataj / np.tile(psthj, [1, np.shape(dataj)[1]])
    corr = np.reshape(corr, [1, np.shape(corr)[0], np.shape(corr)[1]], order='F')
    corr = np.tile(corr,[l,1,1])
    corr = np.reshape(corr,[np.shape(corr)[0]*np.shape(corr)[1],np.shape(corr)[2]],order='F')
    psth = np.reshape(psth, [np.shape(psth)[0], 1, ], order='F')
    output = np.tile(psth, [1, np.shape(corr)[1]]) * corr
    output = output[:length, :]

    return(output)


def ccg_v2(ts,idx,events,max_time=1000,event_window=1):
    '''

    :param ts: all spikes
    :param idx: cell ids of all spikes
    :param events: any event, (usually breath on)
    :param max_time:
    :param event_window:
    :return: ccg_out -
    '''

    jitterwindow=25

    sub_ts = ts[ts<max_time]
    events = events[events>5]
    bt,cell_id,bins = proc.bin_trains(ts,idx,max_time=max_time,binsize=0.001)
    T,T_bins = proc.raster2tensor(bt,bins,events,pre=event_window/2,post=event_window/2)
    # triangle function
    n_t = T.shape[0]
    t = np.arange(-(n_t-1),(n_t-1))
    theta = n_t-np.abs(t)
    del t
    NFFT = int(nextpow2(2*n_t))
    target = np.array([int(i) for i in NFFT/2+np.arange((-n_t+2),n_t)])

    FR = np.mean(bt,axis=1)*1000

    n_unit = FR.shape[0]

    ncomparisons = int((n_unit**2 - n_unit)/2)+1
    # Preallocate for time
    ccg_out = np.zeros([len(theta),ncomparisons])
    raw_ccg = np.zeros_like(ccg_out)

    # Calculate CCG for all cross correlations
    count = -1
    for ii in tqdm(np.arange(n_unit-1)): # V1 cell
        t1 = T[:, ii, :]
        for jj in np.arange(ii+1,n_unit):  # V2 cell
            t2 = T[:, jj, :]
            count+=1
            if FR[ii]>2 and FR[jj]>2:


                # ccg =np.mean(scipy.signal.fftconvolve(t1,t2),axis=1)

                ccg = xcorrfft(t1,t2,NFFT)
                ccg = np.squeeze(np.nanmean(ccg[:, target], axis=0))

                tt1 = jitter(t1,jitterwindow)
                tt2 = jitter(t2,jitterwindow)
                tempjitter = xcorrfft(tt1,tt2,NFFT)
                tempjitter = np.squeeze(np.nanmean(tempjitter[:, target], axis=0))
                ccg_out[:,count] = (ccg - tempjitter) / np.multiply(np.sqrt(FR[ii] * FR[jj]), theta)
                raw_ccg[:,count] = ccg / np.multiply(np.sqrt(FR[ii] * FR[jj]), theta)

    return(ccg_out,raw_ccg)


def get_ccg_peaks(corrected_ccg,thresh = 7):
    '''
    Assumes ccg is in milliseconds
    :param corrected_ccg:
    :return:
    '''

    centerpt = int(np.ceil(corrected_ccg.shape[0]/2))-1
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

    return(exc_connx,inh_connx)

