#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_SEED=987324
export EOS_ANALYSIS_INFO=1

export EOS_MCMC_BURN_IN=500
export EOS_MCMC_PROPOSAL='cauchy'
export EOS_MCMC_SAMPLES=100000
export EOS_MCMC_UPDATE_SIZE=1000

main $@
