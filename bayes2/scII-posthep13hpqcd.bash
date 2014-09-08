#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=30000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CLUSTERS=30
export PMC_CHUNKSIZE=2000
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 200"

export PMC_NUMBER_OF_JOBS=200
export LL_QUEUE=serial
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=60
#export PMC_CLIENT_ARGV="--resume-samples --step 3"

# initial value
export GOF_MODE_0="{ 3.62530304e+00   8.16975209e-01   8.12508650e-01   2.25407457e-01
   1.28596053e-01   3.77803136e-01   1.27853110e+00   4.18948919e+00
   2.26603217e-01   3.79864788e-01  -4.77334534e+00   8.54547166e-01
   1.09399709e+00   2.84451023e-01   1.94163468e-01   2.76087255e-01
  -1.54327242e+00   6.72847512e-03  -2.17553476e-02  -4.20036088e-03
   9.66285133e-01   1.03642034e+00   1.07314700e+00   1.06611989e+00
   3.14737684e-01  -2.22669617e+00  -1.03645322e-02  -7.75933960e-01
   4.62804915e-01   3.50468240e-01 }"

main $@
