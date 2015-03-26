#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

# export EOS_UNC_EXTRA_ARGS=''
export EOS_UNC_OBSERVABLE='B_q->ll::BR@Untagged,q=s,l=mu'
export EOS_UNC_INPUT="$BASE_NAME/sm_unc_Kstar/mcmc_pre_merged.hdf5"
export EOS_UNC_PARAMETER='Re{c10} -4.18 -4.1 20'

main $@
