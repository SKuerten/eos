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

export UNC_SAMPLES=100000
export UNC_WORKERS=1

# just priors and theory constraints
export UNC_PMC_INPUT="
    --pmc-input ${BASE_NAME}/sm_uncFFhpqcd/pmc_monolithic.hdf5 0 100000
    --pmc-sample-directory /data/final
"

# as above + experimental constraints
export UNC_PMC_INPUT="
    --pmc-input ${BASE_NAME}/../pmc_sm_posthep13hpqcd.hdf5 0 100000
"

export PMC_NUMBER_OF_JOBS=1
export PMC_UNCERTAINTY_INPUT="${BASE_NAME}/sm_uncFFhpqcd/pmc_monolithic.hdf5"
#export PMC_CLIENT_ARGV="--n-samples 500"
export LL_QUEUE=test
export LL_FINAL_QUEUE=${LL_QUEUE}
export PMC_POLLING_INTERVAL=30

main $@