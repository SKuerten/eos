#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CHUNKSIZE=2000
export PMC_GROUP_BY_RVALUE=2
export PMC_CLUSTERS=30
export PMC_INITIALIZATION="$PMC_INITIALIZATION --pmc-r-value-no-nuisance 0"
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 50"
export PMC_SKIP_INITIAL=0.2

export PMC_NUMBER_OF_JOBS=150
export PMC_RESOURCE_MANAGER=loadleveler
export LL_QUEUE=test
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=30

main $@
