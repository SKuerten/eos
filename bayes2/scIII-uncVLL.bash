#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export UNC_PMC_INPUT="
    --pmc-input /gpfs/work/pr85tu/ru72xaf2/eos/2013-fall/2014-09-26/scIII_uncVLL/A.hdf5 0 20
"

export PMC_NUMBER_OF_JOBS=200
# EOS can't deal with files merged with h5py and pytables
export PMC_UNCERTAINTY_INPUT="/gpfs/work/pr85tu/ru72xaf2/eos/2013-fall/2014-09-26/scIII_uncVLL/D.hdf5"
#export PMC_CLIENT_ARGV="--n-samples 40"
export PMC_RESOURCE_MANAGER=loadleveler
export LL_QUEUE=test
export LL_FINAL_QUEUE=${LL_QUEUE}
export PMC_POLLING_INTERVAL=2

main $@
