#Usage: sh batch_run_kilosort.sh </path/to/mousedir> <path_filter>


pload=$1
p_spec=$2
first='yes'
for subdir in ${pload}/${p_spec}/
do
    fn=$subdir/*.imec*.ap.bin
    if [ ${first} = 'yes' ];then
        echo ${fn}
        echo  "first"
        jobid=$(qsub -v fn=${fn} ./run_kilosort.sh)
        echo ${jobid}
        

    first='no'
else


    echo ${fn}
    sleep 0.1
    old_job=${jobid}
    jobid=$(qsub -W depend=afterany:${old_job} -v fn=${fn} ./run_kilosort.sh)
    echo ${jobid}
    fi
done




