#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=1000
export MCMC_PRERUN_CHAINS=12
export MCMC_PRERUN_PARALLEL=1

export PMC_FINAL_CHUNKSIZE=2000000
export PMC_CLUSTERS=50

export PMC_NUMBER_OF_JOBS=100

## 2013-08-06
## six groups (11 2 10 10 11 4)
# group  chain                                      sign     MAP    in chain
# 0      00,01,03,05,09,14,23,25,34,37,42           (+,-,+)  126.04    01
# 1      02,44                                      (+,-,-)   81.54    02
# 2      04,10,13,20,30,31,32,40,45,47
# 3      06,07,15,18,21,24,27,35,39,46
# 4      08,11,12,16,19,28,29,33,38,41,43
# 5      17,22,26,36
## best fit in chain 01 (log posterior = 126.04)
##"{ +0.51935 -4.62045 +3.97579 +0.81016 +0.22576 +0.14828 +0.36790 +1.29185 +4.16181 +0.22756 +0.36936 -4.68417 +0.79522 +0.95757 +0.21450 -0.22727 +0.14025 -0.84457 +0.06380 +0.15777 +0.02846 +0.97311 +0.96438 +1.07620 +1.03309 +0.49411 +0.34048 }"

export GOF_MODE_0="{ +0.51935 -4.62045 +3.97579 +0.81016 +0.22576 +0.14828 +0.36790 +1.29185 +4.16181 +0.22756 +0.36936 -4.68417 +0.79522 +0.95757 +0.21450 -0.22727 +0.14025 -0.84457 +0.06380 +0.15777 +0.02846 +0.97311 +0.96438 +1.07620 +1.03309 +0.49411 +0.34048 }"
export GOF_MODE_1=
export GOF_MODE_2=
export GOF_MODE_3=
export GOF_MODE_4=
export GOF_MODE_5=

# just let them die out in first step
# export PMC_IGNORE_GROUPS="
#     --pmc-ignore-group 0
#     --pmc-ignore-group 1
#     --pmc-ignore-group 3
#     --pmc-ignore-group 4
# "

main $@
