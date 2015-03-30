#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=20000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CHUNKSIZE=2000
export PMC_GROUP_BY_RVALUE=2
export PMC_CLUSTERS=30
export PMC_INITIALIZATION="$PMC_INITIALIZATION --pmc-r-value-no-nuisance 0"
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 50"
export PMC_SKIP_INITIAL=0.2

export PMC_NUMBER_OF_JOBS=150
export PMC_RESOURCE_MANAGER=loadleveler
export LL_QUEUE=serial
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=30
export PMC_CLIENT_ARGV=

# from MCMC
export GOF_MODE_0="{ +0.8021108650711575 +0.2251039979593224 +0.1610123300237060 +0.3807146884221629 +1.2657927475281121 +4.1860944601087953 +0.2299838076860519 +0.3614550700867759 -4.7700659133068406 +0.8946216873175491 +1.4044354165670023 +0.2735628441051813 +0.5994305806783029 +0.2634049647952009 -1.0452578908031900 -0.4453366089310697 +0.1582595879232034 -0.2877400953206614 +0.8526850103059546 +0.7407427660155119 +1.3059499318277168 +1.2455020707236528 +0.3070952091434586 -2.8770144757088207 -0.2457427016284675 -0.7049425677259025 +0.3882791614229382 +0.3504606840782825 }"

# BOBYQA
export GOF_MODE_0="{
 +0.8055901835040782 +0.2253215695660005 +0.1533769215876475
 +0.3723333148919589 +1.2721523752692474 +4.1900458502138829
 +0.2295760961040862 +0.3635645481913719 -4.7881265687600374
 +0.8899565747711397 +1.4008444856245303 +0.2761829386484402
 +0.6424293303275985 +0.2643509477120786 -1.2192582222432056
 -0.4453201556071481 -0.4054688760473760 -0.2955671226134940
 +0.8465996367745171 +0.7314378302165920 +1.3140745858302991
 +1.2385063468294117 +0.3081952067558230 -2.8632544791109784
 -0.2532389697409660 -0.7087540075099961 +0.3897113504411607
 +0.3499840289169792 }"
main $@