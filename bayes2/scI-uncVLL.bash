#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export PMC_NUMBER_OF_JOBS=500
export PMC_UNCERTAINTY_INPUT="${BASE_NAME}/../2014-09-18/scI_posthep13/pmc_parameter_samples_4.hdf5_merge"
#export PMC_CLIENT_ARGV="--n-samples 50"
export PMC_RESOURCE_MANAGER=loadleveler
export LL_QUEUE=test
export LL_FINAL_QUEUE=${LL_QUEUE}
export PMC_POLLING_INTERVAL=2

main $@
