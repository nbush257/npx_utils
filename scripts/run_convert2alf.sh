#PBS -l mem=16
#PBS -l nodes=1:ppn=1
#PBS -l walltime=8:00:00
#PBS -P a113801f-349d-42b5-9490-aa02e70721e3
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N convert2alf
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs

source activate iblenv
cd /active/ramirez_j/ramirezlab/nbush/projects/npx_utils/scripts
python convert2alf.py /active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-28/ -b