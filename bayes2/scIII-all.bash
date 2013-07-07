#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=1000
export MCMC_PRERUN_CHAINS=12
export MCMC_PRERUN_PARALLEL=1

export PMC_MONO_FINAL_CHUNKSIZE=120000
#export PMC_MONO_IGNORE_GROUPS="
#    --pmc-ignore-group 0
#    --pmc-ignore-group 3
#    --pmc-ignore-group 4
#"

#export GOF_MODE_0="{ -0.35554 -3.45311 +4.08030 +0.79575 +0.22477 +0.13375 +0.44117 +1.26285 +4.24831 +0.22488 +0.31470 -4.87291 +0.94112 +1.03693 +0.20420 -0.69308 +0.20531 -0.48351 -0.16742 +0.01527 +0.02980 +0.97040 +0.86936 +0.57132 +1.13274 +0.33613 -1.81887 +0.09969 +0.71174 +0.46536 +0.35230 }"
#export GOF_MODE_1="{ +0.50059 +2.81001 -3.81000 +0.80938 +0.22447 +0.11035 +0.37307 +1.25305 +4.26033 +0.23075 +0.29796 -4.51913 +0.97427 +1.07101 +0.24393 +0.37965 +0.56752 +3.19164 +0.15139 +0.14162 -0.10614 +1.05544 +1.17875 +0.56225 +0.95269 +0.33200 -2.00987 -0.06628 -0.53872 +0.47567 +0.35379 }"
#export GOF_MODE_2="{ -0.34780 +3.87766 -4.76606 +0.80093 +0.22496 +0.15038 +0.40864 +1.29634 +4.26695 +0.22925 +0.39947 -4.21832 +0.79115 +0.90924 +0.22926 +0.14063 +0.22253 -0.58458 -0.06882 +0.08320 -0.08429 +0.95464 +0.96586 +1.30550 +0.97400 +0.31655 -2.15198 -0.03669 -0.69669 +0.45329 +0.35561 }"
#export GOF_MODE_3="{ +0.50139 -5.13577 +4.27248 +0.80166 +0.22492 +0.16047 +0.37186 +1.25612 +4.26114 +0.22572 +0.34878 -4.64481 +0.84717 +1.07761 +0.27826 +1.24381 +0.60753 +4.29465 +0.07096 -0.06664 -0.05688 +1.15994 +0.90073 +1.10748 +1.03456 +0.32825 -2.03591 -0.10788 +0.83116 +0.44360 +0.35400 }"

main $@
