from .. import proc
import click


@click.command()
@click.argument('ni_bin_fn',help='path to binary nidq file')
@click.argument('pleth',help='Analog Channel index for pleth')
@click.argument('dia',help='Analog Channel index for diaphragm')
@click.argument('opto',help='Analog Channel index for optogenetics')
def main(ni_bin_fn,):
    #TODO: create csvs and data files for the integrated data
    # We are making this a CLI because it takes forever to run, and we want
    # to be able to load in the integrated trace.
    pass