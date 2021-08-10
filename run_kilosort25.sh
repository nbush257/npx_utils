#PBS -P Opioid
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N kilosort
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs

# USAGE:
# qsub -v p=</path/to/bin> run_kilosort.sh



cd $PROJ/npx_utils/cli
echo ${p}
cfg=/active/ramirez_j/ramirezlab/nbush/helpers/kilosort2/configFiles/cfg_neb_20201118.m
matlab -nodesktop -nosplash -r "run_ks2_NEB('$cfg','$p')"




