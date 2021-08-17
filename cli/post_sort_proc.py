'''
Given a directory of the Kilosort results, this function uses the IBL libraries to compute metrics on the sorted neurons.
Potentially - can be rerun after manual curation
Potentially- pipes into Tprime and Cwaves
'''

import ibllib
import sys
import os