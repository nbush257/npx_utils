#PBS -l mem=16gb
#PBS -l nodes=1:ppn=2
#PBS -l walltime=14:00:00
#PBS -P Opioid
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N standard_postproc
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs
#PBS -J 0-42

source activate opioid
export PYTHONWARNINGS="ignore"

dir_list=()
t_max_list=()
opto_len_list=()
data_org_fn=/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/ks3_dirs.csv
while IFS=, read -r ks3_dir t_max opto_len
do
    dir_list+=($ks3_dir)
    t_max_list+=($t_max)
    opto_len_list+=($opto_len)

done < $data_org_fn;


cd $PROJ/npx_utils/cli
ii=$PBS_ARRAY_INDEX
echo Working on ${dir_list[$ii]}
python -W ignore standard_postproc.py ${dir_list[$ii]} --t_max ${t_max_list[$ii]} --stim_len ${opto_len_list[$ii]}



#for ii in {0..50}
#do
    #echo Working on ${dir_list[$ii]}
    #python -W ignore standard_postproc.py ${dir_list[$ii]} --t_max ${t_max_list[$ii]} --stim_len ${opto_len_list[$ii]}
#
#done






