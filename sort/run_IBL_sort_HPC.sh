#PBS -l mem=64gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -l walltime=24:00:00
#PBS -P c1d84aeb-cb9d-45c9-8be2-698a685d6739
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N IBL_SORT
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs


module load GCC cuda libxcb
source activate pyks2
cd /active/ramirez_j/ramirezlab/nbush/projects/npx_utils/sort

python ibl_sort_mouse.py /archive/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/raw/mi-20220919-03
python ibl_sort_mouse.py /archive/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/raw/mi-20220919-04

