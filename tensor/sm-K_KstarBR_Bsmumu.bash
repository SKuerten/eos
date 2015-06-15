#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_SEED=987324
export EOS_ANALYSIS_INFO=0

export EOS_MCMC_BURN_IN=
export EOS_MCMC_COVARIANCE="Kstar-FF-cov.txt"
export EOS_MCMC_INITIAL_VALUES="uniform"
export EOS_MCMC_PROPOSAL='gauss'
export EOS_MCMC_SAMPLES=500000
export EOS_MCMC_UPDATE_SIZE=500

main $@
