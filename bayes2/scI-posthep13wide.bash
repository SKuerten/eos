#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=1000
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=1

export PMC_FINAL_CHUNKSIZE=1000000
export PMC_CLUSTERS=35

export PMC_NUMBER_OF_JOBS=100

## 2013-08-27
## three groups (12 7 1)
# group  chains                                     sign     MAP    in chain
# 0      00,05,07,10,12,13,14,15,16,17,18,19        (+,-,+)  610.87 10
# 1      01,03,04,06,08,09,11                       (-,+,-)  609.30 11
# 2      02                                         (+,-,+)  603.92 02

## best fit in chain 10 (log posterior = 610.87)
export GOF_MODE_0="{ +0.51636 -5.37917 +4.27465 +0.80462 +0.22493 +0.15322 +0.35446 +1.26614 +4.18183 +0.22485 +0.34098 -4.49045 +0.86200 +1.26545 +0.24202 +0.26504 +0.20295 -1.58977 +0.21311 -0.44415 +0.19267 +0.72127 +1.25625 +1.41559 +0.86345 +0.32533 -1.94655 -0.39455 +1.63107 +0.47934 +0.34891 }"
export GOF_MODE_1="{ -0.33534 +4.02786 -4.39547 +0.82187 +0.22476 +0.10898 +0.36229 +1.28522 +4.19316 +0.22835 +0.39763 -4.94967 +0.80822 +1.47542 +0.23278 +0.29552 +0.20785 +0.90569 +0.05202 -0.34075 -0.25430 +0.80939 +1.11115 +1.22313 +0.65547 +0.32333 -2.03507 -0.45145 -1.08912 +0.51807 +0.35148 }"

# just let them die out in first step
# export PMC_IGNORE_GROUPS="
#     --pmc-ignore-group 0
#     --pmc-ignore-group 1
#     --pmc-ignore-group 3
#     --pmc-ignore-group 4
# "

main $@
