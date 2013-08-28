#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export PMC_SEED=13614623

export UNC_SAMPLES=100000
export UNC_WORKERS=8

main $@
