#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=1000
export MCMC_PRERUN_CHAINS=12
export MCMC_PRERUN_PARALLEL=1

export PMC_FINAL_CHUNKSIZE=2000000
export PMC_CLUSTERS=35

export PMC_NUMBER_OF_JOBS=100

## 2013-08-04
## five groups (12 11 9 14 2)
# group  chain                                      sign     MAP
# 0      00,02,03,05,06,07,09                       (+,+,-)  577.85
# 1      01,14,19,20,23,25,29,34,37,41,42           (+,-,+)  591.39
# 2      04,08,11,12,22,33,38,39,43                 (-,-,+)  579.12
# 3      10,13,15,16,24,26,28,30,31,32,40,45,46,47  (-,+,-)  595.26
# 4      17,36                                      (-,+,+)  379.45
## best fit in chain 31 (log posterior = 595.2576429)
## "{ -0.35358 +3.91213 -4.50644 +0.78265 +0.22537 +0.14942 +0.37579 +1.28086 +4.20675 +0.22593 +0.39510 -4.74175 +0.83440 +0.98440 +0.23945 +0.38658 +0.59543 +4.84271 +0.08043 +0.00637 -0.03163 +0.99674 +0.98797 +1.15753 +0.96398 +0.32985 -1.99409 +0.12739 -0.69279 +0.46153 +0.35202 }"

export GOF_MODE_0="{ +0.52142 +2.46609 -4.14779 +0.79852 +0.22496 +0.13169 +0.37027 +1.28870 +4.17438 +0.22713 +0.31846 -4.53995 +0.89488 +0.98856 +0.22745 +0.08665 +0.54334 +4.14528 +0.10088 -0.07458 +0.11086 +1.14279 +0.94642 +0.57890 +1.00825 +0.32589 -1.94208 +0.15666 -0.38285 +0.43509 +0.34904 }"
export GOF_MODE_1="{ +0.52408 -5.04488 +4.51675 +0.79443 +0.22465 +0.13358 +0.36134 +1.27543 +4.18612 +0.22714 +0.37454 -4.55776 +0.80679 +0.95565 +0.25634 +0.77683 +0.61380 +4.74385 -0.01748 -0.02585 -0.06435 +0.89974 +1.08147 +1.00833 +0.88796 +0.32033 -2.21177 -0.05754 +0.73591 +0.53563 +0.34962 }"
export GOF_MODE_2="{ -0.35309 -3.64739 +3.57822 +0.80451 +0.22496 +0.13897 +0.35122 +1.28777 +4.17760 +0.22643 +0.33531 -4.31367 +0.91827 +1.06166 +0.20065 -1.09897 +0.53608 +4.59444 -0.06931 +0.10874 -0.05331 +1.05562 +0.94580 +0.55389 +1.06966 +0.34996 -1.62471 +0.07110 +0.71437 +0.46679 +0.34820 }"
export GOF_MODE_3="{ -0.35358 +3.91213 -4.50644 +0.78265 +0.22537 +0.14942 +0.37579 +1.28086 +4.20675 +0.22593 +0.39510 -4.74175 +0.83440 +0.98440 +0.23945 +0.38658 +0.59543 +4.84271 +0.08043 +0.00637 -0.03163 +0.99674 +0.98797 +1.15753 +0.96398 +0.32985 -1.99409 +0.12739 -0.69279 +0.46153 +0.35202 }"
export GOF_MODE_4="{ -0.40224 +3.96294 +4.79657 +0.80094 +0.22516 +0.16259 +0.37560 +1.28169 +4.15913 +0.22749 +0.15339 -4.93906 +1.41607 +1.29206 -0.04977 -0.46512 +0.06077 +1.18450 +0.05324 +0.24233 -0.08896 +1.16369 +0.94509 +1.28611 +1.27887 +0.32906 -1.91707 -0.07012 -0.97789 +0.50118 +0.34973 }"

# just let them die out in first step
# export PMC_IGNORE_GROUPS="
#     --pmc-ignore-group 0
#     --pmc-ignore-group 1
#     --pmc-ignore-group 3
#     --pmc-ignore-group 4
# "

main $@