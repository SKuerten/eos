#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

# export EOS_UNC_EXTRA_ARGS=''
export EOS_UNC_OBSERVABLE='B_q->ll::BR@Untagged,q=s,l=mu'
export EOS_UNC_INPUT='/data/eos/2015-tensor/2015-03-24/sm_unc_K/mcmc_1144.hdf5'
export EOS_UNC_PARAMETER='Re{c10} -4.18 -4.1 10'

main $@
