#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=1000
export MCMC_PRERUN_CHAINS=12
export MCMC_PRERUN_PARALLEL=1

export PMC_FINAL_CHUNKSIZE=1000000
export PMC_CLUSTERS=35
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 200"

export PMC_NUMBER_OF_JOBS=200
export PMC_RESOURCE_MANAGER=loadleveler
export LL_QUEUE=test
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=30

export GOF_MODE_0="{ -0.342938 3.94893 -4.61573 0.808068 0.225358 0.128712 0.375286 1.27586 4.18304 0.226087 0.376615 -4.82848 0.85512 1.07168 0.239454 0.356338 0.220514 -0.9973 -3.8297e-05 -0.00197059 -0.000421926 0.869076 1.0222 1.08701 0.997858 0.310198 -2.34839 -0.0267642 -0.682554 0.449699 0.349935 }"
export GOF_MODE_1="{ 0.505892 -5.00182 4.50871 0.807911 0.225357 0.127915 0.374663 1.27451 4.18482 0.226373 0.360349 -4.85599 0.837452 1.02942 0.23867 0.332993 0.221718 -1.01975 -0.000501089 -0.00082091 -0.00216783 0.855726 1.01414 1.10662 0.972481 0.305353 -2.43659 -0.0324392 0.778763 0.448027 0.350032 }"

main $@
