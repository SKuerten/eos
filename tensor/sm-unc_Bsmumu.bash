#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_SEED=1243879

export EOS_ANALYSIS_INFO=0

export EOS_MCMC_BURN_IN=2000
export EOS_MCMC_INTEGRATION_POINTS=16
export EOS_MCMC_PROPOSAL='gauss'
export EOS_MCMC_SAMPLES=500000
export EOS_MCMC_SCALE_REDUCTION=5
export EOS_MCMC_UPDATE_SIZE=500

# export EOS_UNC_EXTRA_ARGS=''
export EOS_UNC_OBSERVABLE='B_q->ll::BR@Untagged,q=s,l=mu'
export EOS_UNC_INPUT="$BASE_NAME/sm_unc_Kstar/mcmc_pre_merged.hdf5"
export EOS_UNC_PARAMETER='Re{c10} -4.18 -4.1 5'

main $@
