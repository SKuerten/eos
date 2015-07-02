#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_SEED=237548
export EOS_ANALYSIS_INFO=1

export EOS_MCMC_INITIAL_VALUES="fixed"
export EOS_MCMC_PROPOSAL='gauss'
export EOS_MCMC_SAMPLES=150000
export EOS_MCMC_UPDATE_SIZE=500

main $@
