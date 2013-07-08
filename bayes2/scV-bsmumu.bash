#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=10000
export MCMC_PRERUN_CHAINS=4
export MCMC_PRERUN_PARALLEL=1

export PMC_NUMBER_OF_JOBS=3
export SGE_FINAL_QUEUE=short
export PMC_POLLING_INTERVAL=5

main $@
