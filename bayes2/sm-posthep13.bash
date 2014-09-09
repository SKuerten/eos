#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=20000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CHUNKSIZE=1500
export PMC_GROUP_BY_RVALUE=2
export PMC_FINAL_CHUNKSIZE=1000000
export PMC_CLUSTERS=25
export PMC_INITIALIZATION="$PMC_INITIALIZATION --pmc-r-value-no-nuisance 0"
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 50"
export PMC_SKIP_INITIAL=0.2

export PMC_NUMBER_OF_JOBS=150
export PMC_RESOURCE_MANAGER=loadleveler
export LL_QUEUE=test
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=30

# starting points
export GOF_MODE_0="{
   8.16939207e-01   2.25423898e-01   1.31954842e-01   3.75220033e-01
   1.27566624e+00   4.18569906e+00   2.27072271e-01   3.83180816e-01
  -4.72611010e+00   8.44426225e-01   1.13481492e+00   2.42191380e-01
   2.83244844e-01   2.23841849e-01  -8.90081345e-01  -3.81471526e-04
  -2.41470705e-03   5.96079746e-03   8.64071526e-01   1.01376357e+00
   1.10214143e+00   1.08311367e+00   3.16252920e-01  -2.23821276e+00
  -3.47933659e-02  -7.45934009e-01   4.49064923e-01   3.50099828e-01 }"

# 8.16990764e-01   2.25423713e-01   1.32093141e-01   3.76146848e-01
#    1.27663458e+00   4.18577316e+00   2.27059427e-01   3.83139209e-01
#   -4.71933052e+00   8.44267564e-01   1.13463705e+00   2.41724025e-01
#    2.77917518e-01   2.23074636e-01  -9.00197969e-01  -4.44703902e-04
#   -2.50619374e-03   6.09836464e-03   8.64881818e-01   1.01369090e+00
#    1.10041577e+00   1.08291863e+00   3.16170385e-01  -2.23932457e+00
#   -3.43276574e-02  -7.43369978e-01   4.48400747e-01   3.50090993e-01 }"

main $@