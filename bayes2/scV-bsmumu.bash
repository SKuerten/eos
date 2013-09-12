#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=1000
export MCMC_PRERUN_CHAINS=4
export MCMC_PRERUN_PARALLEL=1

export PMC_PATCH_LENGTH=20
export PMC_CLUSTERS=35
export PMC_CHUNKSIZE=300
export PMC_FINAL_CHUNKSIZE=50000

export PMC_NUMBER_OF_JOBS=3
export SGE_FINAL_QUEUE=short
export PMC_POLLING_INTERVAL=5

export PMC_RESOURCE_MANAGER=loadleveler
export LL_QUEUE=test
export LL_FINAL_QUEUE=${LL_QUEUE}

main $@
