#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_SEED=7442

export EOS_ANALYSIS_INFO=0

export EOS_MCMC_ACCEPTANCE_MAX=0.35
export EOS_MCMC_BURN_IN=
export EOS_MCMC_INITIAL_VALUES="fixed"
export EOS_MCMC_INTEGRATION_POINTS=16
export EOS_MCMC_PROPOSAL='gauss'
export EOS_MCMC_SAMPLES=100000
# export EOS_MCMC_SCALE_NUISANCE=
# export EOS_MCMC_SCALE_REDUCTION=1
export EOS_MCMC_UPDATE_SIZE=500

main $@