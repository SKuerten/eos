#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_ANALYSIS_INFO=0

export EOS_MCMC_BURN_IN=500
export EOS_MCMC_INTEGRATION_POINTS=16
export EOS_MCMC_PROPOSAL='gauss'
export EOS_MCMC_SAMPLES=50000
export EOS_MCMC_SCALE_REDUCTION=5
export EOS_MCMC_UPDATE_SIZE=500

main $@
