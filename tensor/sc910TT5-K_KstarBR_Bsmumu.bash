#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_ANALYSIS_INFO=0

export EOS_IS_SEED=13461346
export EOS_IS_SAMPLES=200000

export EOS_MCMC_SCALE_REDUCTION=5
export EOS_MCMC_BURN_IN=500
export EOS_MCMC_UPDATE_SIZE=500
export EOS_MCMC_SAMPLES=40000
export EOS_MCMC_PROPOSAL='gauss'

export EOS_VB_COMPONENTS_PER_GROUP=3
#export EOS_VB_EXTRA_OPTIONS='--indices 0 3'
#export EOS_VB_INPUT='/data/eos/2014-tensor/2014-12-02/sc910TT5_K_KstarBR_Bsmumu/mcmc_pre_merged.hdf5'
export EOS_VB_R_VALUE=5
export EOS_VB_REL_TOL=0.00000001
export EOS_VB_SKIP_INITIAL=0.05
export EOS_VB_THIN=40

main $@
