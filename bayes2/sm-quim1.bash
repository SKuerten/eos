#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=20000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_CHUNKSIZE=1500
export PMC_CLUSTERS=25
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 50"
export PMC_FINAL_CHUNKSIZE=1000000
export PMC_GROUP_BY_RVALUE=2
export PMC_INITIALIZATION="$PMC_INITIALIZATION --pmc-r-value-no-nuisance 0"

# starting point
export GOF_MODE_0="{ +0.83000 +0.22491 +0.14264 +0.36603 +1.26252 +4.20406 +0.22833 +0.35492 -4.84499 +0.86519 +1.02284 +0.18174 +0.36779 +0.30492 -1.68015 -0.08616 -0.03118 -0.04175 +1.10085 +1.01359 +1.02285 +0.94492 +0.42387 +0.36085 }"

# result
export GOF_MODE_0="{ 0.816376 0.22542 0.131316 0.375315 1.27266 4.19299 0.227377 0.322071 -4.79421 0.963722 1.01747 0.186921 0.44118 0.312312 -1.20656 0.000745103 -0.00129701 0.000677836 0.985773 1.01908 0.999938 0.980015 0.448642 0.34914 }" # log(posterior) = 153.907437

main $@
