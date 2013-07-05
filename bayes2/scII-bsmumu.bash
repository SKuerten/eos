#!/bin/bash
#vim: set sts=4 et :

source job.bash

export MCMC_PRERUN_SAMPLES=10000
export MCMC_PRERUN_CHAINS=4
export MCMC_PRERUN_PARALLEL=1

main $@
