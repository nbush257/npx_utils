#Usage: sh batch_run_kilosort.sh </path/to/mousedir> <path_filter>
# Use a wildcard in <path filter. before and after the runname. eg: *m2020-20*



pload=$1
p_spec=$2
first='yes'
for subdir in ${pload}/${p_spec}
do
    p=$(ls -d $subdir/${p_spec}/)
    echo $p
    p=${p::-1}
    if [ ${first} = 'yes' ];then
        echo  "first"
        jobid=$(qsub -v p=${p} ./run_kilosort25.sh)
        echo ${jobid}
        first='no'
    else
        sleep 0.1
        old_job=${jobid}
        jobid=$(qsub -W depend=afterany:${old_job} -v p=${p} ./run_kilosort25.sh)
        echo ${jobid}
    fi
done




