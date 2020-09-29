import click
import sys
sys.path.append('../')
import os
import readSGLX
import data
from pathlib import Path
import pandas as pd
import numpy as np


@click.command()
@click.argument('ni_bin_fn')
@click.argument('chans',type=int,nargs=-1)
@click.option('-s','--sub',type=int,default=20)
@click.option('-l','--save_loc',default=None)
def extract_and_subsamp(ni_bin_fn,chans,sub,save_loc):
   if save_loc is None:
      save_loc = os.path.split(ni_bin_fn)[0]
   df = pd.DataFrame()
   for chan_id in chans:
      XA = data.get_ni_analog(ni_bin_fn,chan_id)
      meta = readSGLX.readMeta(Path(ni_bin_fn))
      sr = readSGLX.SampRate(meta)
      XA_sub = XA[::sub]

      df[f'XA{chan_id}'] = XA_sub.astype('float32')

   tmax = len(XA) / sr
   tvec = np.arange(0, tmax, 1 / sr)
   tvec_sub = tvec[::sub]
   df['t'] = tvec_sub.astype('float32')
   df.set_index('t',inplace=True)

   df.to_csv(os.path.join(save_loc,'analog_subsampled.csv'))

if __name__== '__main__':
   extract_and_subsamp()

