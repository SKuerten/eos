#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_MCMC_SCALE_REDUCTION=3
export EOS_MCMC_BURN_IN=200
export EOS_MCMC_UPDATE_SIZE=100
export EOS_MCMC_SAMPLES=500

main $@
