#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=30000
export MCMC_PRERUN_UPDATE_SIZE=750
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_FINAL_CHUNKSIZE=1000000
export PMC_CLUSTERS=35

export PMC_NUMBER_OF_JOBS=100

export GOF_MODE_0="{
 -0.358615  3.440986 -4.032573  0.801017  0.225559  0.124655  0.37629
  1.298014  4.196732  0.229584  0.348315 -4.641066  0.889469  1.042933
  0.236632  0.79614   0.206562 -1.090807 -0.042693  0.022172 -0.134255
  0.981034  0.908701  1.106006  0.922608  0.504033  0.351572 }"

export GOF_MODE_1="{ +0.52439 -3.75594 +4.51645 +0.79923 +0.22531 +0.12333 +0.38377 +1.24702 +4.17836 +0.22498 +0.33406 -5.10448 +0.86162 +0.97931 +0.24440 +0.51273 +0.21893 +0.22203 -0.10823 +0.01728 -0.02408 +1.04434 +0.95864 +1.05523 +0.93491 +0.48974 +0.36620 }"

main $@
