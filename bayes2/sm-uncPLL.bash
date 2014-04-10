#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=15000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=4
export MCMC_PRERUN_PARALLEL=1

export PMC_SEED=13614623

export PMC_CHUNKSIZE=500
export PMC_CLUSTERS=15
export PMC_FINAL_CHUNKSIZE=100000
export PMC_GROUP_BY_RVALUE=1.5
export PMC_INITIALIZATION="$PMC_INITIALIZATION --pmc-r-value-no-nuisance 0"

export UNC_SAMPLES=100000
export UNC_WORKERS=6
export UNC_PMC_INPUT="
    --pmc-input ${BASE_NAME}/sm_uncPLL/pmc_monolithic.hdf5 0 ${UNC_SAMPLES}
    --pmc-sample-directory /data/final
"

main $@
