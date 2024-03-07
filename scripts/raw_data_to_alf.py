'''
reformat our raw data to match IBL-like ALF standards.
WIP!!
'''

from ibllib.pipes import misc
import spikeglx
from pathlib import Path
import re
import shutil
iblscripts_folder = Path(r'D:/iblscripts')
local_subjects_path = Path(r'D:\test\Subjects')
remote_subjects_path = Path(r'D:\remote_test\Subjects')
session_path = Path(r'D:\test\Subjects\m2024-03\m2024-03_g0')

# This replaces the "imec" folder with "probe"
def hack_probe_folders(session_path):
    def _get_probe_number(probe_path):
        probe_num = int(re.search('(?<=imec)\d',probe_path.name).group())
        out_str = f'probe{probe_num:02.0f}'
        return(out_str)

    ephys_files = spikeglx.glob_ephys_files(session_path)
    for efi in ephys_files:
        if 'ap' not in efi:
            continue
        else:
            probe_path_orig = efi['ap'].parent
            probe_path_dest = probe_path_orig.parent.joinpath(_get_probe_number(probe_path_orig))
            shutil.move(probe_path_orig,probe_path_dest)

            


hack_probe_folders(session_path)
# ephys_files = spikeglx.glob_ephys_files(session_path)
misc.create_ephyspc_params()

misc.create_basic_transfer_params(local_data_path=local_subjects_path,remote_data_path=remote_subjects_path)
# misc.create_custom_ephys_wirings(iblscripts_folder)
misc.rename_ephys_files(session_path)
misc.move_ephys_files(session_path)


misc.multi_parts_flags_creation(session_path)
misc.probe_labels_from_session_path(session_path)

misc.create_alyx_probe_insertions(str(session_path))