#PBS -l ngpus=1
#PBS -l mem=32gb
#PBS -l walltime=04:00:00
#PBS -P a113801f-349d-42b5-9490-aa02e70721e3
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N Kilosort3
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs
#PBS -J 0-205

cd /active/ramirez_j/ramirezlab/nbush/projects/dynaresp/dynaresp

data_org_fn=/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/probe_list.csv # This will need to point to the probe list
dir_list=()
while IFS=, read -r probe_dir probe_dir_win mouse_id
do
    dir_list+=($probe_dir)

done < $data_org_fn;

ii=$PBS_ARRAY_INDEX
fn=${dir_list[$ii]}
echo Working on $fn

matlab -nodesktop -nosplash -r "run_ks3_HPC('$fn')"


