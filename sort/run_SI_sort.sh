#PBS -l mem=64gb
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -l walltime=12:00:00
#PBS -P a113801f-349d-42b5-9490-aa02e70721e3
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N SI_SORT
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs


module load GCC cuda libxcb FFTW
source activate pyks2
cd /active/ramirez_j/ramirezlab/nbush/projects/npx_utils/sort

python spikeinterface_pyks.py /gpfs/home/nbush/si_scratch/m2023-24

