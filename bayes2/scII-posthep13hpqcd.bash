#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=30000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CLUSTERS=30
export PMC_CHUNKSIZE=2000
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 200"

export PMC_NUMBER_OF_JOBS=32
export LL_QUEUE=shorttest
export LL_FINAL_QUEUE=shorttest
export PMC_POLLING_INTERVAL=30
export PMC_CLIENT_ARGV="--resume-samples --step 3"

main $@
