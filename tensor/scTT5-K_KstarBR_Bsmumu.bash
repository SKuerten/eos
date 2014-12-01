#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_MCMC_SCALE_REDUCTION=5
export EOS_MCMC_BURN_IN=500
export EOS_MCMC_UPDATE_SIZE=500
export EOS_MCMC_SAMPLES=40000
export EOS_MCMC_PROPOSAL='cauchy'

main $@
