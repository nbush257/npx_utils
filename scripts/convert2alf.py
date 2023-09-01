from pathlib import Path
from atlaselectrophysiology.extract_files import extract_data
import click

DEBUG = False

def run_one(ks_path,ephys_path,out_path):
    ks_path = Path(ks_path)
    param_file = ks_path.joinpath('params.py')
    if not param_file.exists():
        print('No params file found. Probably not a valid phy folder')
        return(-1)

    if ephys_path is None:
        ephys_path = ks_path.parent
    
    if out_path is None:
        out_path = ks_path.parent.joinpath('alf')
    
    if not out_path.exists():
        out_path.mkdir()


    print(f'Loading kilosort from:\n\t {ks_path}')
    print(f'Loading ephys from:\n\t {ephys_path}')
    print(f'saving to alf:\n\t {out_path}')
    if DEBUG:
        print('Not running because DEBUG is True')
    else:
        extract_data(ks_path, ephys_path, out_path)
    return(0)

def run_batch(run_path):
    run_path = Path(run_path)
    params_files = list(run_path.glob('*/*/*ks2/params.py'))
    params_files.sort()

    if len(params_files)==0:
        raise ValueError("No phy folders found.")

    for fn in params_files:
        imec_path = fn.parent
        if ('orig' in str(imec_path)) or ('alf' in str(imec_path)):
            print(f'Skipping\n\t{imec_path}')
            params_files.remove(fn)


    print('Found the following phy folders:')
    [print(x) for x in params_files]
    for fn in params_files:
        imec_path = fn.parent
        run_one(imec_path,ephys_path=None,out_path=None)
        
@click.command()
@click.argument('in_path',type=str)
@click.option('--ephys_path','-e',help='path to raw ephys',default=None,type=str)
@click.option('--out_path','-o',help = 'path to save the alf data',default=None,type=str)
@click.option('--batch','-b',help='use switch if converting all from a run',is_flag=True)
def main(in_path,ephys_path,out_path,batch):
    if batch:
        run_batch(in_path)
    else:
        run_one(in_path,ephys_path,out_path)

    

if __name__=='__main__':
    main()

