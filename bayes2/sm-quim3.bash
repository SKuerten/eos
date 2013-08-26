#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=20000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

# more components as there are two modes, but chains can see both
export PMC_CHUNKSIZE=1500
export PMC_CLUSTERS=50
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 20"
export PMC_FINAL_CHUNKSIZE=1000000
export PMC_GROUP_BY_RVALUE=1.5
export PMC_INITIALIZATION="$PMC_INITIALIZATION --pmc-r-value-no-nuisance 0"

main $@
