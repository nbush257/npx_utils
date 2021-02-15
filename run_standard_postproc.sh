#PBS -l mem=16gb
#PBS -l nodes=1:ppn=4
#PBS -l walltime=14:00:00
#PBS -P Opioid
#PBS -m abe
#PBS -M nicholas.bush@seattlechildrens.org
#PBS -N standard_postproc
#PBS -q paidq
#PBS -e /active/ramirez_j/ramirezlab/nbush/logs
#PBS -o /active/ramirez_j/ramirezlab/nbush/logs
#PBS -J 0-18

source activate opioid

f_list=(
#/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2020-20/catgt_m2020-20_g0/m2020-20_g0_imec0/imec0_ks3
#/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2020-20/catgt_m2020-20_g1_g0/m2020-20_g1_g0_imec0/imec0_ks3
#/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2020-20/catgt_m2020-20_g1_g1/m2020-20_g1_g1_imec0/imec0_ks3
#/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2020-20/catgt_m2020-20_g1_g2/m2020-20_g1_g2_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2020-23/catgt_m2020-23_g0/m2020-23_g0_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2020-23/catgt_m2020-23_g1/m2020-23_g1_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2020-23/catgt_m2020-23_g2/m2020-23_g2_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-01/catgt_m2021-01_g0/m2021-01_g0_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-01/catgt_m2021-01_g1/m2021-01_g1_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-01/catgt_m2021-01_g2/m2021-01_g2_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-01/catgt_m2021-01_g3/m2021-01_g3_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-04/catgt_m2021-04_g0/m2021-04_g0_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-04/catgt_m2021-04_g1/m2021-04_g1_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-04/catgt_m2021-04_g2/m2021-04_g2_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-04/catgt_m2021-04_g3/m2021-04_g3_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-05/catgt_m2021-05_g0/m2021-05_g0_imec0/imec0_ks3
/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2021-05/catgt_m2021-05_g1/m2021-05_g1_imec0/imec0_ks3
)

cd $PROJ/npx_utils/cli
ff=${f_list[${PBS_ARRAY_INDEX}]}
echo $ff
python standard_postproc.py $ff




