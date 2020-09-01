#Usage: sh batch_run_kilosort.sh /path/to/mousedir


pload=$1
first='yes'
for subdir in ${pload}/*/
do
    fn=$subdir/*.imec*.ap.bin
    if [ ${first} = 'yes' ];then
        echo ${fn}
        echo  "first"
        jobid=$(qsub -v fn=${fn} ./run_kilosort.sh)
        echo ${jobid}
        
    fi    
    first='no'


    echo ${fn}
    sleep 0.1
    old_job=${jobid}
    jobid=$(qsub -W depend=afterany:${old_job} -v fn=${fn} ./run_kilosort.sh)
done




