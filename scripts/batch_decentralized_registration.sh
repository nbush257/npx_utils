#PBS -l ngpus=1
#PBS -l mem=16gb
#PBS -l walltime=72:00:00
#PBS -P a113801f-349d-42b5-9490-aa02e70721e3
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N decentralized_reg
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs
#PBS -J 0-205

module load cuda90/toolkit/9.2.148
source activate decentralized_reg

cd /active/ramirez_j/ramirezlab/nbush/projects/npx_utils/cli

data_org_fn=/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/probe_list.csv # This will need to point to the probe list
dir_list=()
while IFS=, read -r probe_dir probe_dir_win mouse_id
do
    dir_list+=($probe_dir)

done < $data_org_fn;

ii=$PBS_ARRAY_INDEX
echo Working on ${dir_list[$ii]}
python run_decentralized_registration.py ${dir_list[$ii]}

#python run_decentralized_registration.py /active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/test/catgt_m2021-10_g2/m2021-10_g2_imec0

