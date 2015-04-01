source ${EOS_SCRIPT_PATH}/constraints.bash
source ${EOS_SCRIPT_PATH}/predictions.bash
source ${EOS_SCRIPT_PATH}/priors.bash
source ${EOS_SCRIPT_PATH}/scan.bash

export EOS_ANALYSIS_INFO=1
export EOS_SEED=12345

non_empty() {
    local var=$1
    if [[ -z ${!var} ]]; then
        echo "No $1 given"
        exit -1
    fi
}

export EOS_IS_INPUT=  # default: $output_dir/vb.hdf5
export EOS_IS_INTEGRATION_POINTS=32

gof() {
    scenario=$1; shift
    data=$1;shift

    gof_index=${1:-0}; shift
    non_empty "gof_index"

    scan=SCAN_${scenario}
    constraints=CONSTRAINTS_${data}
    nuisance=NUISANCE_${data}
    mode=GOF_MODE_${gof_index}

    eos-scan-mc \
        --debug \
        --goodness-of-fit ${!mode} \
        ${!scan} \
        ${!nuisance} \
        ${!constraints} \
        --output $output_dir/gof_${idx}.hdf5 \
        > $output_dir/gof_${idx}.log 2>&1
}

is() {
    step=${1}
    shift

    local input=$EOS_IS_INPUT
    if [[ -z $input ]]; then
        input=$output_dir/vb.hdf5
    fi

    local output=$EOS_IS_OUTPUT
    if [[ -z $output ]]; then
        output=$input
    fi

    seed=$(expr $EOS_SEED "+" 642134)

    ../py-eos/is.py \
        --analysis-info $EOS_ANALYSIS_INFO \
        --eos-integration-points $EOS_IS_INTEGRATION_POINTS \
        --input $input \
        --output $output \
        --samples $EOS_IS_SAMPLES \
        --seed $seed \
        --step $step \
        > $output_dir/is_$step.log 2>&1
}

export EOS_MCMC_BURN_IN=
export EOS_MCMC_INTEGRATION_POINTS=16
export EOS_MCMC_SAMPLES=40000
export EOS_MCMC_UPDATE_SIZE=1000
export EOS_MCMC_SCALE_NUISANCE=1
export EOS_MCMC_SCALE_REDUCTION=1
export EOS_MCMC_PROPOSAL='gauss'
mcmc() {
    prerun_index=${1}
    non_empty "prerun_index"
    seed=$(expr $EOS_SEED "+" ${prerun_index} "*" 1000)

    ../py-eos/mcmc.py \
        --analysis-info $EOS_ANALYSIS_INFO \
        --burn-in $EOS_MCMC_BURN_IN \
        --eos-integration-points $EOS_MCMC_INTEGRATION_POINTS \
        --output $output_dir/mcmc_${prerun_index}.hdf5 \
        --proposal $EOS_MCMC_PROPOSAL \
        --samples $EOS_MCMC_SAMPLES \
        --scale-nuisance $EOS_MCMC_SCALE_NUISANCE \
        --scale-reduction $EOS_MCMC_SCALE_REDUCTION \
        --seed $seed \
        --update-after $EOS_MCMC_UPDATE_SIZE \
        > $output_dir/mcmc_${prerun_index}.log 2>&1
}

export EOS_OPT_MAXEVAL=5000
export EOS_OPT_MAXEVAL_LOCAL=$EOS_OPT_MAXEVAL
export EOS_OPT_TOL=1e-14
export EOS_OPT_TOL_LOCAL=$EOS_OPT_TOL

opt() {
    gof_index=${1:-0}; shift

    # set default if arg not given
    nlopt_algorithm=${1:-"LN_BOBYQA"}; shift
    non_empty "nlopt_algorithm"

    local_alg=${1}; shift

    EOS_MODE=GOF_MODE_${gof_index}

    ../py-eos/optimize.py --algorithm $nlopt_algorithm --local-algorithm $local_alg \
        --initial-guess ${!EOS_MODE} \
        --max-evaluations "${EOS_OPT_MAXEVAL}" --tolerance "${EOS_OPT_TOL}" \
        --max-evaluations-local "${EOS_OPT_MAXEVAL_LOCAL}" --tolerance-local "${EOS_OPT_TOL_LOCAL}"
        > $output_dir/opt_${nlopt_algorithm}_${local_alg}_${gof_index}.log 2>&1
}

opt_multi () {
    first_mode_index=${1:-0}; shift

    second_mode_index=${1}; shift
    if [[ -z ${second_mode_index} ]] ; then
        second_mode_index=first_mode_index
    fi

    algorithms="LN_BOBYQA LN_COBYLA"
    # for i in $(seq -f %01.0f ${first_mode_index} ${second_mode_index}); do
    for ((i=first_mode_index; i <= second_mode_index; i++)); do
        for alg in $algorithms; do
            echo "Running $alg for mode $i"
            opt $i $alg &
        done
    done
}

export EOS_VB_COMPONENTS_PER_GROUP=15
export EOS_VB_EXTRA_OPTIONS=
export EOS_VB_INIT_METHOD="random"
export EOS_VB_IS_INPUT=
export EOS_VB_PRUNE=
export EOS_VB_MCMC_INPUT=
export EOS_VB_SKIP_INITIAL=0.05
export EOS_VB_THIN=100
export EOS_VB_REL_TOL=
export EOS_VB_R_VALUE=2

vb() {

    local input_mode=${1}
    shift

    local input_file=

    case ${input_mode} in
        mcmc)
            input_file="$output_dir/mcmc_pre_merged.hdf5"
            if [[ -f $EOS_VB_MCMC_INPUT ]]; then
                input_file=$EOS_VB_MCMC_INPUT
            fi
            ;;
        is)
            input_file="$output_dir/vb.hdf5"
            if [[ -f $EOS_VB_IS_INPUT ]]; then
                input_file=$EOS_VB_IS_INPUT
            fi
            ;;
        *)
            echo "Invalid input mode '"${input_mode}"' given! Choose 'mcmc' or 'is'!"
            exit -1
            ;;
    esac
    non_empty "input_file"
    local input_arg="--${input_mode}-input ${input_file}"

    step=${1}
    shift
    non_empty "step"

    output=${1}
    shift
    if [[ -z $output ]]; then
        if [[ $step == "0" ]]; then
            output="$output_dir/vb.hdf5"
        else
            output='APPEND'
        fi
    fi

    non_empty "output"

    seed=$(expr $EOS_SEED "+" 654198)
    ../py-eos/vb.py \
        --analysis-info $EOS_ANALYSIS_INFO \
        --components-per-group $EOS_VB_COMPONENTS_PER_GROUP \
        --init-method $EOS_VB_INIT_METHOD \
        ${input_arg} \
        --output $output \
        --prune $EOS_VB_PRUNE \
        --rel-tol $EOS_VB_REL_TOL \
        --R-value $EOS_VB_R_VALUE \
        --seed $seed \
        --step $step \
        --skip-initial $EOS_VB_SKIP_INITIAL \
        --thin $EOS_VB_THIN \
        $EOS_VB_EXTRA_OPTIONS \
        > $output_dir/vb_${step}.log 2>&1

    # create a backup of this small file
    if [[ ${input_mode} == mcmc ]]; then
        cp "$output" "${output}~"
    fi
}

EOS_UNC_GLOBAL_OPTIONS=",model=WilsonScan,scan-mode=cartesian,form-factors=KMPW2010"
unc() {
    local low=$1; shift
    local high=$1; shift

    ../py-eos/unc.py \
        ${input_arg} \
        --kinematics ${EOS_UNC_KINEMATICS} \
        --input-range $low $high \
        --mcmc-input ${EOS_UNC_INPUT} \
        --observable ${EOS_UNC_OBSERVABLE}${EOS_UNC_GLOBAL_OPTIONS} \
        --output $output_dir/unc_${data}_${low}_${high}.hdf5 \
        --parameter ${EOS_UNC_PARAMETER}
}

## Job Main Function ##
main() {
    local name=${0}

    local name=${name##*/}
    local name=${name%%.*}
    local scenario=${name%%-*}
    local data=${name##*-}

    non_empty "BASE_NAME"

    non_empty "scenario"
    echo "[scenario = ${scenario}]"

    non_empty "data"
    echo "[data = ${data}]"

    EOS_SCAN=SCAN_${scenario}
    export EOS_SCAN=${!EOS_SCAN}
    EOS_NUISANCE=NUISANCE_${data}
    export EOS_NUISANCE=${!EOS_NUISANCE}
    EOS_CONSTRAINTS=CONSTRAINTS_${data}
    export EOS_CONSTRAINTS=${!EOS_CONSTRAINTS}

    export output_dir=${BASE_NAME}/${scenario}_${data}
    echo "[output dir = ${output_dir}]"
    mkdir -p $output_dir

    cmd=${1}
    shift

    case ${cmd} in
        driver)
            out=$output_dir/vb.hdf5
            rm $out
            vb mcmc 0 &&
            is 0 &&
            vb is 1 APPEND &&
            is 1 &&
            vb is 2 APPEND &&
            is 2 &&
            vb is 3 APPEND &&
            is 3 &&
            vb is 4 APPEND &&
            is 4
            ;;
        gof)
            gof ${scenario} ${data} $@
            ;;
        info)
            ../py-eos/analysis_info.py
            ;;
        is)
            is $@
            ;;
        mcmc)
            mcmc $@
            ;;
        noop)
            # do nothing
            ;;
        opt)
            opt $@
            ;;
        opt_multi)
            opt_multi $@
            ;;
        unc)
            unc $@
            ;;
        vb)
            vb $@
            ;;
        *)
            echo "Invalid command ${cmd} given!"
            exit -1
            ;;
    esac
}
