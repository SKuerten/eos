#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=1000
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_FINAL_CHUNKSIZE=1000000
export PMC_CLUSTERS=35
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 200"

export PMC_NUMBER_OF_JOBS=200
export LL_QUEUE=test
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=30

export GOF_MODE_0="{ +3.29252 +1.53127 +0.79322 +0.22453 +0.12248 +0.38200 +1.28059 +4.20313 +0.22683 +0.38742 -4.99852 +0.86879 +1.01986 +0.31315 +0.17005 +0.29202 -1.64686 -0.02520 +0.03199 +0.10932 +0.93765 +1.02632 +1.05994 +1.07650 +0.33636 -1.81446 -0.03319 +0.18443 +0.42910 +0.35035 }"


main $@
