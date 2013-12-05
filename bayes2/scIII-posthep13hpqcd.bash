#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 200"
export PMC_PATCH_LENGTH=250
export PMC_CHUNKSIZE=4000
export PMC_CLUSTERS=50
export PMC_FINAL_CHUNKSIZE=1000000
export PMC_GROUP_BY_RVALUE=2
export PMC_SKIP_INITIAL=0.5

export PMC_NUMBER_OF_JOBS=170
export LL_QUEUE=test
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=30
#export PMC_CLIENT_ARGV="--resume-update --step 0"

export PMC_UNCERTAINTY_INPUT="${BASE_NAME}/scIII_posthep13/pmc_parameter_samples_15.hdf5_merge"

export GOF_MODE_0="{ -0.33141 +3.16815 -4.56588 -0.09423 -0.02942 -0.90439 +0.80155 +0.22552 +0.16202 +0.38175 +1.25927 +4.15169 +0.22789 +0.40183 -4.58468 +0.80333 +1.05876 +0.28428 +0.29458 +0.26898 -1.38199 +0.03550 -0.14248 +0.09251 +0.95082 +0.91995 +1.07713 +0.84737 +0.31491 -2.22826 +0.01830 -0.11836 +0.45370 +0.34887 }"
export GOF_MODE_1="{ +0.52241 -4.82677 +3.65294 +0.03890 -1.95076 -0.81385 +0.80462 +0.22557 +0.12204 +0.42439 +1.28430 +4.19108 +0.22733 +0.34599 -4.71760 +0.83019 +1.08150 +0.31178 +0.69484 +0.32831 -0.88219 -0.03328 -0.01734 +0.02903 +0.94983 +0.98920 +1.11202 +1.08637 +0.34066 -1.82676 +0.20772 +0.52345 +0.48359 +0.35194 }"
export GOF_MODE_2="{ +0.00911 +3.36408 -1.35330 -0.42440 +3.31407 +2.46291 +0.80218 +0.22629 +0.16434 +0.37805 +1.29132 +4.19719 +0.23146 +0.34505 -4.83070 +1.09985 +0.86010 +0.32500 +0.35584 +0.29500 -0.99758 -0.03328 -0.05042 +0.08477 +0.84707 +1.05902 +1.08845 +1.06507 +0.35450 -1.50053 +0.10522 +0.35772 +0.49752 +0.34922 }"
export GOF_MODE_3="{ +0.07285 -3.97195 +2.30836 +0.44874 -3.56296 -2.14964 +0.79967 +0.22566 +0.09698 +0.32026 +1.29718 +4.16893 +0.22953 +0.36479 -4.19963 +1.01618 +0.83667 +0.27388 +0.10975 +0.24163 +1.06587 +0.00964 +0.10977 +0.04998 +0.84022 +1.05413 +0.98118 +0.95430 +0.36176 -1.31237 +0.07887 +0.38731 +0.40667 +0.34820 }"

main $@
