"""
Olympus VSI outputs tiffs in a strange folder structure to preserve metadata. This copies and renames
the actual images to be easily used in sharp-track
"""
import os
import click
import glob
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import scipy.io.matlab as sio
import numpy as np

@click.command()
@click.argument('p_load')
def main(p_load):
    nmperpx = []
    print(p_load)
    for root,dir,files in os.walk(p_load):
        if len(dir)==0:
            print(root)
            try:
                im_file = glob.glob(os.path.join(root,'*.tif'))[0]
                print(f'Moving {im_file}')
                xml_file = glob.glob(os.path.join(root,'*.xml'))[0]
                parts = Path(root).parts
                basename = parts[-2].replace('.vsi.Collection','')
                basename += f'_{parts[-1][-1]}'
                dest = os.path.join(p_load,basename)+'.tiff'
                shutil.copy(im_file,dest)

                xml_dat = ET.parse(xml_file)
                aa = './Groups_0/ImageTemplate/ImageAxis0Resolution'
                bb = './Groups_0/ImageTemplate/ImageAxis1Resolution'
                xscale = xml_dat.find(aa).attrib['value']
                xscale = float(re.search('.*(?:\D|^)(\d+)',xscale).group())
                yscale = xml_dat.find(bb).attrib['value']
                yscale = float(re.search('.*(?:\D|^)(\d+)',yscale).group())
            except:
                print('='*100)
                print('='*100)
                print('bob')
                print('='*100)
                print('='*100)
                print(root)
                xscale = np.nan
                yscale = np.nan

                nmperpx.append([xscale,yscale])
    nmperpx = np.array(nmperpx)

    sio.savemat(os.path.join(p_load,'nm_per_pixel.mat'),{'nmperpx':nmperpx})



if __name__ == '__main__':
    main()
