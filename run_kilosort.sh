#PBS -l ngpus=1
#PBS -l mem=16gb
#PBS -l walltime=04:00:00
#PBS -P Opioid
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N kilosort
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs

# USAGE:
# qsub -v fn=</path/to/ap.bin> run_kilosort.sh


source activate opioid

cd $PROJ/npx_utils/cli
echo ${fn}
python sort_spikeinterface.py ${fn}




