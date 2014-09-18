#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=9
export MCMC_PRERUN_UPDATE_SIZE=3
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

main $@
