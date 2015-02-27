#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_ANALYSIS_INFO=0

export EOS_IS_SAMPLES=10000

export EOS_MCMC_SCALE_REDUCTION=5
export EOS_MCMC_BURN_IN=200
export EOS_MCMC_UPDATE_SIZE=500
export EOS_MCMC_SAMPLES=50000
export EOS_MCMC_PROPOSAL='cauchy'

export EOS_VB_COMPONENTS_PER_GROUP=3
export EOS_VB_EXTRA_OPTIONS='--indices 0 2 --rel-tol 0.000001'
export EOS_VB_MCMC_INPUT='/data/eos/2014-tensor/2015-01-15/scSP_Bsmumu/mcmc_pre_merged.hdf5'
#export EOS_VB_IS_INPUT='/data/eos/2014-tensor/2014-12-12/scTT5_FH/is.hdf5'
export EOS_VB_PRUNE=100
export EOS_VB_R_VALUE=2
export EOS_VB_SKIP_INITIAL=0.05
export EOS_VB_THIN=50

main $@
