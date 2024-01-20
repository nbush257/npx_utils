#PBS -l mem=64gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -l walltime=16:00:00
#PBS -P a113801f-349d-42b5-9490-aa02e70721e3
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N IBL_SORT
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs


module load GCC cuda libxcb FFTW
source activate pyks2
cd /active/ramirez_j/ramirezlab/nbush/projects/npx_utils/sort

# cp /archive/ramirez_j/ramirezlab/nbush/projects/iso_npx/data/raw/m2023-23 /gpfs/home/nbush/si-scratch

python ibl_sort_mouse.py /gpfs/home/nbush/si_scratch/m2023-23/m2023-23_g0 -S
# python ibl_sort_mouse.py /archive/ramirez_j/ramirezlab/nbush/projects/iso_npx/data/raw/m2023-26
# python ibl_sort_mouse.py /archive/ramirez_j/ramirezlab/nbush/projects/iso_npx/data/raw/m2023-30
# python ibl_sort_mouse.py /archive/ramirez_j/ramirezlab/nbush/projects/iso_npx/data/raw/m2023-30 # This one needs to get run still
# python ibl_sort_mouse.py /archive/ramirez_j/ramirezlab/nbush/projects/iso_npx/data/raw/m2023-32
# python ibl_sort_mouse.py /archive/ramirez_j/ramirezlab/nbush/projects/iso_npx/data/raw/m2023-33
# python ibl_sort_mouse.py /archive/ramirez_j/ramirezlab/nbush/projects/iso_npx/data/raw/m2023-35
# python ibl_sort_mouse.py /archive/ramirez_j/ramirezlab/nbush/projects/iso_npx/data/raw/m2023-36

