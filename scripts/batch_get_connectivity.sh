#PBS -l mem=32
#PBS -l nodes=1:ppn=4
#PBS -l walltime=24:00:00
#PBS -P Opioid
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N get_connectivity
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs
#PBS -J 0-200

#0-200
source activate opioid
export PYTHONWARNINGS="ignore"

dir_list=()
t_max_list=()
opto_len_list=()
data_org_fn=/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/gate_list.csv
while IFS=, read -r gate_dir #t_max
do
    dir_list+=($gate_dir)
#    t_max_list+=($t_max)

done < $data_org_fn;


cd $PROJ/npx_utils/scripts
ii=$PBS_ARRAY_INDEX
echo Working on ${dir_list[$ii]}
python -W ignore get_connectivity.py ${dir_list[$ii]} --tf 1200



