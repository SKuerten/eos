#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

PMC_SEED=64231

UNC_SAMPLES=100000
UNC_WORKERS=8

main $@
