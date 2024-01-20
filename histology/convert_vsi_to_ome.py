'''
Uses the openmicroscopy.org bftools to convert from vsi to OME tiff files. 
Takes only the first series (the high-res images). Sorts the files by creation date and then saves them to a 
"ome" folder that is parallel to the directory where the .vsi images live.
'''
import os
import subprocess
from pathlib import Path
import click
import sys
if sys.platform == 'linux':
    BFCONVERT_PATH = '/active/ramirez_j/ramirezlab/nbush/helpers/bftools/bfconvert'
else:
    BFCONVERT_PATH = r"Y:/helpers/bftools/bfconvert.bat"

@click.command()
@click.argument('vsi_path')
@click.option('--series','-s',default=1,type=str)
@click.option('--bigtiff','-b',is_flag=True)
def main(vsi_path,series,bigtiff):
    load_path = Path(vsi_path)
    save_path = load_path.parent.joinpath('ome')
    vsi_list = list(load_path.glob('*.vsi'))
    vsi_list.sort(key=os.path.getctime)
    if not save_path.exists():
        save_path.mkdir()
    for vsi in vsi_list:
        save_fn = save_path.joinpath(vsi.stem).with_suffix('.ome.tif')

        if sys.platform=='linux':
            if bigtiff:
                run_command = ['sh',BFCONVERT_PATH,vsi,'-series',series,'-overwrite','-bigtiff',save_fn]
            else:
                run_command = ['sh',BFCONVERT_PATH,vsi,'-series',series,'-overwrite',save_fn]
        else:
            if bigtiff:
                run_command = [BFCONVERT_PATH,vsi,'-series',series,'-overwrite','-bigtiff',save_fn]
            else:
                run_command = [BFCONVERT_PATH,vsi,'-series',series,'-overwrite',save_fn]
        subprocess.run(run_command)

if __name__=='__main__':
    main()

