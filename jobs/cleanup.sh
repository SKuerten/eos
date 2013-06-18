#! /bin/bash

if [ "$#" -lt 2 ]; then
  echo Usage: $0 '[qdel | PREFIX STEP [ --zip-log  | --delete | --delete-job-log | --delete-update-log ]]'
  exit
fi

if [ "$1" == "qdel" ]; then
    for ((i=$2; i <= $3 ; i++))
      do
      qdel $i
    done

    exit
fi

PREFIX="sc1_$1"
STEP=$2

LOG_FILES="${PREFIX}_?.log ${PREFIX}_??.log ${PREFIX}_???.log"
JOB_FILES="${PREFIX}_job_${STEP}_?.sh ${PREFIX}_job_${STEP}_??.sh ${PREFIX}_job_${STEP}_???.sh"
HDF5_STEP_FILES="${PREFIX}_job_${STEP}_?.hdf5 ${PREFIX}_job_${STEP}_??.hdf5 ${PREFIX}_job_${STEP}_???.hdf5"
UPDATE_LOG_FILES="${PREFIX}_update_*.log"

if [ "$3" == "--zip-all-log" ]; then
    tar -cjf ${PREFIX}_job.log.tar.bz $LOG_FILES
    tar -cjf ${PREFIX}_update.log.tar.bz $UPDATE_LOG_FILES
elif [ "$3" == "--zip-updates" ]; then
    echo "Doing nothing"
#    tar -cjf ${PREFIX}.sh.tar.bz $JOB_FILES
    # tar -cjf ${PREFIX}.hdf5.tar.bz $HDF5_STEP_FILES
elif [ "$3" == "--delete" ]; then
    rm  $JOB_FILES $HDF5_STEP_FILES
elif [ "$3" == "--delete-job-log" ]; then
    rm $LOG_FILES
elif [ "$3" == "--delete-update-log" ]; then
    rm $UPDATE_LOG_FILES
else
    echo "Would operate on
          `ls -m $JOB_FILES $HDF5_STEP_FILES $LOG_FILES`"
fi
