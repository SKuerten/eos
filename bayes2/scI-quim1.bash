#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=30000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 50"
export PMC_CHUNKSIZE=2500
export PMC_FINAL_CHUNKSIZE=1000000
export PMC_CLUSTERS=35

# focus on SM-like solution as quim did
# 0=B,1=A,2=-++,3=+--
export PMC_IGNORE_GROUPS="
    --pmc-ignore-group 0
    --pmc-ignore-group 2
    --pmc-ignore-group 3
"

export PMC_NUMBER_OF_JOBS=400
export LL_QUEUE=test
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=30

# starting point from mcmc
export GOF_MODE_0="{ -0.3351379314411688 +2.3992576297740249 -4.0851769598215899 +0.8157173266469840 +0.2257253259467213 +0.0988644569421045 +0.3431676119556774 +1.2788896526195150 +4.2051873827008404 +0.2301063253337835 +0.3106148522612496 -4.2150310953864683 +0.9853120849463477 +0.9770369028505587 +0.2755512973088628 -0.1928761606507041 +0.2587397862152883 -0.7915165207309663 -0.0960736871865311 +0.0676048188171190 -0.1221825627334702 +0.9978406380435488 +0.9829243045080486 +0.9124951968954507 +1.0649539820189857 +0.4101969160140702 +0.3492653315437140 }"

# BOBYQA
export GOF_MODE_0="{
 -0.3323645523003657 +2.4459292531830337 -4.0715824095941651
 +0.8162132882893555 +0.2256704914597703 +0.0978404051011350
 +0.3464031223843349 +1.2775630868612597 +4.2000000046947070
 +0.2295443592661097 +0.3150781458671706 -4.2859395265098756
 +0.9865216449345027 +0.9856665230082823 +0.2675586036026610
 -0.1433465379787876 +0.2479007627051550 -0.6772040238931696
 -0.0696082897255431 +0.0563394233194069 +0.0293365428047795
 +1.0042890314787261 +1.0042379598631879 +0.9398500700317864
 +1.0270375202238855 +0.4328348008438499 +0.3509030913366950 }"

main $@
