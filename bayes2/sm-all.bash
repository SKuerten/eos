#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=25000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_FINAL_CHUNKSIZE=1000000

export GOF_MODE_0="{ +0.80093 +0.22496 +0.15038 +0.40864 +1.29634 +4.26695 +0.22925 +0.39947 -4.21832 +0.79115 +0.90924 +0.22926 +0.14063 +0.22253 -0.58458 -0.06882 +0.08320 -0.08429 +0.95464 +0.96586 +1.30550 +0.97400 +0.31655 -2.15198 -0.03669 -0.69669 +0.45329 +0.35561 }"

main $@
