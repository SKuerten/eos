#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=4000
export MCMC_PRERUN_CHAINS=12
export MCMC_PRERUN_PARALLEL=1

export PMC_MONO_FINAL_CHUNKSIZE=120000

main $@
