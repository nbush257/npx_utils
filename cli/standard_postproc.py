try:
    from ..npx_utils import proc as proc
except:
    pass
import proc
import data
import visualize as viz
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

try:
    import npx_utils.proc as proc
except:
    import proc

def main(ks2_dir,ni_bin_fn,PLETH_CHAN=0,DIA_CHAN=1,OPTO_CHAN=2):

    # (DONE) Plot long time-scale raster of all neurons with respiratory rate
    # (DONE) Plot short timescale raster of all neurons with pleth and dia integrated
    # (DONE) Plot respiratory rate and heart rate
    # Plot phase tuning and event psth of dia on for each neuron individually. Include gasp tuning (?) Include apnea tuning (?)
    # Plot event triggered raster of several example breaths
    # Plot raster/psth of opto-triggered respones.
    # Organize plots into a single image
    # Plot depth map of phase tuning.
    # Plot depth map of Tuning depth
    # Plot depth map of tagged neurons
    # Plot Dia/pleth and onsets to verify the detections
    # Compute tagged boolean for all neuron
    # Compute recording depth, preferred phase, modulation depth, and is-modulated for all neurons
    # Save Unit level data in csv

    # NEEDS: ks2dir,ni_binary, epoch times + labels, channels

    # ========================= #
    #  LOAD DATA #
    # ========================= #
    # Need to process in line for memory reasons
    plt.style.use('seaborn-paper')
    sr = data.get_sr(ni_bin_fn)
    spikes = data.get_concatenated_spikes(ks2_dir)
    spikes = data.filter_by_spikerate(spikes,500)
    dia = data.get_ni_analog(ni_bin_fn,DIA_CHAN)
    dia_df,phys_df,dia_int = proc.proc_dia(dia,sr)
    pleth = data.get_ni_analog(ni_bin_fn,PLETH_CHAN)
    pleth_df = proc.proc_pleth(pleth,sr)
    tvec = np.arange(0,len(dia)/sr,1/sr)

    # Compute Phase
    b,a = scipy.signal.butter(4,25/sr/2,btype='low')
    pleth_f = scipy.signal.filtfilt(b,a,pleth)
    phi = proc.calc_phase(pleth_f)
    phi = proc.shift_phi(phi,pleth_df['on_samp'])

    # Plot raster summaries
    f,ax = viz.plot_raster_summary(spikes,phys_df)
    viz.plot_raster_example(spikes,sr,pleth,dia_int,500,5)

    # Plot single cell summaries

    epochs = np.arange(0,1501,300)
    viz.plot_cell_summary_over_time(spikes,100,epochs,dia_df,phi,sr)












