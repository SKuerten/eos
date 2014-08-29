#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=30000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_FINAL_CHUNKSIZE=1000000
export PMC_CLUSTERS=35
export PMC_CHUNKSIZE=2000
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 200"

export PMC_NUMBER_OF_JOBS=200
export LL_QUEUE=test
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=30

# initial value
export GOF_MODE_0="{ 3.55945 1.03523 0.822831 0.225263 0.123673 0.362218 1.27049 4.17963 0.227291 0.376693 -4.81637 0.853348 1.04057 0.293706 0.512071 0.28712 0.024897 -0.0367197 0.0150743 -0.00716304 0.924349 1.05859 1.0605 1.09032 0.290425 -2.74669 -0.00654406 -0.536591 0.4525 0.349738 }"

main $@
