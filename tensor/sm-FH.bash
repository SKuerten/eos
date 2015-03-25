#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_UNC_KINEMATICS='s_min 1 s_max 6'
export EOS_UNC_OBSERVABLE='B->Kll::F_Havg@LargeRecoil,q=u,l=mu'
export EOS_UNC_INPUT="/gpfs/work/pr85tu/ru72xaf2/eos/2015-tensor/2015-03-25/sm_unc_K/mcmc_pre_merged.hdf5"
export EOS_UNC_PARAMETER='Re{cT} -0.1 -0.1 2'

main $@
