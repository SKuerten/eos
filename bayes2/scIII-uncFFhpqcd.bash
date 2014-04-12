#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=15000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=4
export MCMC_PRERUN_PARALLEL=1

export PMC_SEED=13614623

export PMC_CHUNKSIZE=1000
export PMC_CLUSTERS=35
export PMC_FINAL_CHUNKSIZE=100000
export PMC_GROUP_BY_RVALUE=1.5
export PMC_INITIALIZATION="$PMC_INITIALIZATION --pmc-r-value-no-nuisance 0"

export UNC_SAMPLES=250000
export UNC_WORKERS=1
export UNC_PMC_INPUT="
    --pmc-input ${BASE_NAME}/scIII_posthep13hpqcd/pmc_scIII_posthep13hpqcd_D.hdf5 0 ${UNC_SAMPLES}
"

main $@