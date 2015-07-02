#!/bin/bash

# uncertainty propagation: select observable and scenario via command
# line instead of script name. This job file is meant to be used
# standalone!

source ${EOS_SCRIPT_PATH}/unc_scenario.bash
source ${EOS_SCRIPT_PATH}/unc_observable.bash
source ${EOS_SCRIPT_PATH}/job.bash

export EOS_UNC_INTEGRATION_POINTS=${EOS_UNC_INTEGRATION_POINTS:-16}
unc() {
    local input=$1; shift; non_empty input
    local low=$1; shift; non_empty low
    local high=$1; shift; non_empty high

    ../py-eos/unc.py \
        --eos-integration-points ${EOS_UNC_INTEGRATION_POINTS} \
        --fix ${EOS_UNC_FIX} \
        --input-range $low $high \
        --kinematics ${EOS_UNC_KINEMATICS} \
        --input $input \
        --observable ${EOS_UNC_OBSERVABLE} \
        --output $output_dir/unc_${low}_${high}.hdf5 \
        --parameter ${EOS_UNC_PARAMETER}
}

non_empty "BASE_NAME"

scenario=$1; shift; non_empty scenario
observable=$1; shift; non_empty observable

EOS_UNC_FIX=UNC_FIX_${scenario}
export EOS_UNC_FIX=${!EOS_UNC_FIX}
if [[ -z ${EOS_UNC_FIX} ]]; then
    echo "Unknown scenario: '$scenario'"
    exit -1
fi
EOS_UNC_PARAMETER=UNC_PAR_${scenario}
export EOS_UNC_PARAMETER=${!EOS_UNC_PARAMETER}

EOS_UNC_OBSERVABLE=UNC_OBS_${observable}
export EOS_UNC_OBSERVABLE=${!EOS_UNC_OBSERVABLE}
if [[ -z ${EOS_UNC_OBSERVABLE} ]]; then
    echo "Unknown observable: '$observable'"
    exit -1
fi
EOS_UNC_KINEMATICS=UNC_KIN_${observable}
export EOS_UNC_KINEMATICS=${!EOS_UNC_KINEMATICS}

export output_dir=${BASE_NAME}/${scenario}_${observable}
echo "[output dir = ${output_dir}]"
mkdir -p $output_dir

unc $@
