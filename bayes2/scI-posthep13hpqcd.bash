#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_FINAL_CHUNKSIZE=1000000
export PMC_CLUSTERS=35
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 200"

export PMC_NUMBER_OF_JOBS=100
export PMC_RESOURCE_MANAGER=loadleveler
export LL_QUEUE=test
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=30

export GOF_MODE_0="{ -0.343786 3.8715 -4.26496 0.807785 0.225356 0.128685 0.375282 1.27612 4.18303 0.226953 0.378157 -4.76926 0.851243 1.08989 0.266115 0.0699375 0.264589 -1.91838 0.00381722 -0.0445109 -0.00627076 0.956856 1.02041 1.10673 1.02105 0.330055 -2.03729 -0.0501991 -0.631509 0.450115 0.349936 }"
export GOF_MODE_1="{ 0.50596 -4.84777 4.16572 0.807719 0.225356 0.127809 0.374676 1.27457 4.18521 0.227166 0.369809 -4.85056 0.818255 1.03005 0.26617 0.0633233 0.266968 -1.8996 0.0041643 -0.0422793 -0.00563943 0.94086 1.01032 1.11479 0.973953 0.326265 -2.10365 -0.0662236 0.722301 0.448382 0.350055 }"

main $@
