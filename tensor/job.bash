source ${EOS_SCRIPT_PATH}/constraints.bash
source ${EOS_SCRIPT_PATH}/predictions.bash
source ${EOS_SCRIPT_PATH}/priors.bash
source ${EOS_SCRIPT_PATH}/scan.bash

export EOS_ANALYSIS_INFO=1

non_empty() {
    local var=$1
    if [[ -z ${!var} ]]; then
        echo "No $1 given"
        exit -1
    fi
}

export EOS_MCMC_BURN_IN=
export EOS_INTEGRATION_POINTS=16
export EOS_MCMC_SAMPLES=40000
export EOS_MCMC_UPDATE_SIZE=1000
export EOS_MCMC_SCALE_NUISANCE=1
export EOS_MCMC_SCALE_REDUCTION=1
export EOS_MCMC_PROPOSAL='gauss'
mcmc() {
    scenario=${1}
    shift

    data=${1}
    shift

    prerun_index=${1}
    non_empty "prerun_index"
    seed=$(expr 12345 "+" ${prerun_index} "*" 1000)

    scan=SCAN_${scenario}
    constraints=CONSTRAINTS_${data}
    nuisance=NUISANCE_${data}

    output_dir=${BASE_NAME}/${scenario}_${data}
    mkdir -p $output_dir
    ../py-eos/mcmc.py \
        --analysis-info $EOS_ANALYSIS_INFO \
        --burn-in $EOS_MCMC_BURN_IN \
        --eos-integration-points $EOS_INTEGRATION_POINTS \
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
    scenario=${1}
    shift

    data=${1}
    shift

    gof_index=${1}
    non_empty "gof_index"
    shift

    nlopt_algorithm=${1}
    non_empty "nlopt_algorithm"
    shift

    local_alg=${1}
    shift

    EOS_MODE=GOF_MODE_${gof_index}

    mkdir -p ${BASE_NAME}/${scenario}_${data}

    ../py-eos/optimize.py --algorithm $nlopt_algorithm --local-algorithm $local_alg \
        --initial-guess ${!EOS_MODE} \
        --max-evaluations "${EOS_OPT_MAXEVAL}" --tolerance "${EOS_OPT_TOL}" \
        --max-evaluations-local "${EOS_OPT_MAXEVAL_LOCAL}" --tolerance-local "${EOS_OPT_TOL_LOCAL}"
        > ${BASE_NAME}/${scenario}_${data}/py_opt_${nlopt_algorithm}_${local_alg}_${gof_index}.log 2>&1
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

    cmd=${1}
    shift

    case ${cmd} in
        mcmc)
            mcmc ${scenario} ${data} $@
            ;;
        merge)
            mcmc_merge ${scenario} ${data} $@
            ;;
        pmc)
            pmc_monolithic ${scenario} ${data} $@
            ;;
        info)
            ../py-eos/analysis_info.py
            ;;
        gof)
            gof ${scenario} ${data} $@
            ;;
        opt)
            opt ${scenario} ${data} $@
            ;;
        unc)
            unc ${scenario} ${data} $@
            ;;
        *)
            echo "No command given!"
            exit -1
            ;;
    esac
}
