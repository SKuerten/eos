#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=30000
export MCMC_PRERUN_UPDATE_SIZE=750
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 50"
export PMC_CHUNKSIZE=2500
export PMC_FINAL_CHUNKSIZE=1000000
export PMC_CLUSTERS=35

export PMC_NUMBER_OF_JOBS=100

export GOF_MODE_0="{ -0.347206672851063 3.56041715690668 -3.87090642647645 0.790892807523567 0.226010988021726 0.130198166951617 0.402444241905088 1.24625838462123 4.19478739977751 0.228807458085115 0.349600926438918 -4.54675221636615 0.89506272157188 1.06340023212004 0.231254674635201 0.832280580119816 0.206149412391413 0.107052629117302  0.0672921643632544 0.00215031621959379 0.0190226500533313 1.03620349338661 0.980238281232099 0.853692478371726 0.937869490947532 0.39600199511099 0.3591305818016 }"

export GOF_MODE_1="{ +0.52439 -3.75594 +4.51645 +0.79923 +0.22531 +0.12333 +0.38377 +1.24702 +4.17836 +0.22498 +0.33406 -5.10448 +0.86162 +0.97931 +0.24440 +0.51273 +0.21893 +0.22203 -0.10823 +0.01728 -0.02408 +1.04434 +0.95864 +1.05523 +0.93491 +0.48974 +0.36620 }"

main $@
