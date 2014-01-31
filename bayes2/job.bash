# vim: set sts=4 et :

## MCMC Prerun ##
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=1000
export MCMC_PRERUN_SCALE_REDUCTION=10
export MCMC_PRERUN_PARALLEL=1

source ${EOS_SCRIPT_PATH}/constraints.bash
source ${EOS_SCRIPT_PATH}/predictions.bash
source ${EOS_SCRIPT_PATH}/priors.bash
source ${EOS_SCRIPT_PATH}/scan.bash

mcmc_pre() {
    scenario=${1}
    if [[ -z ${scenario} ]] ; then
        echo "No scenario given!"
        exit -1
    fi
    echo "[scenario = ${scenario}]"

    shift
    data=${1}
    if [[ -z ${data} ]] ; then
        echo "No data given!"
        exit -1
    fi
    echo "[data = ${data}]"

    shift
    idx=${1}
    if [[ -z ${idx} ]] ; then
        echo "No prerun index given!"
        exit -1
    fi
    seed=$(expr 12345 "+" ${idx} "*" 1000)

    scan=SCAN_${scenario}
    constraints=CONSTRAINTS_${data}
    nuisance=NUISANCE_${data}

    mkdir -p ${BASE_NAME}/${scenario}_${data}
    eos-scan-mc \
        --seed ${seed} \
        --debug \
        --parallel ${MCMC_PRERUN_PARALLEL} \
        --prerun-chains-per-partition ${MCMC_PRERUN_CHAINS} \
        --prerun-min ${MCMC_PRERUN_SAMPLES} \
        --prerun-max ${MCMC_PRERUN_SAMPLES} \
        --prerun-update ${MCMC_PRERUN_UPDATE_SIZE} \
        --prerun-only \
        --proposal "MultivariateGaussian" \
        --scale-reduction ${MCMC_PRERUN_SCALE_REDUCTION} \
        --store-prerun \
        ${!constraints} \
        ${!scan} \
        ${!nuisance} \
        --output "${BASE_NAME}/${scenario}_${data}/mcmc_pre_${idx}.hdf5" \
        > ${BASE_NAME}/${scenario}_${data}/mcmc_pre_${idx}.log 2>&1
}

mcmc_merge() {
    echo
}

hc() {
    scenario=${1}
    if [[ -z ${scenario} ]] ; then
        echo "No scenario given!"
        exit -1
    fi
    echo "[scenario = ${scenario}]"

    shift
    data=${1}
    if [[ -z ${data} ]] ; then
        echo "No data given!"
        exit -1
    fi
    echo "[data = ${data}]"

    seed=789654

    scan=SCAN_${scenario}
    constraints=CONSTRAINTS_${data}
    nuisance=NUISANCE_${data}

    mkdir -p ${BASE_NAME}/${scenario}_${data}
    eos-scan-mc \
        --seed ${seed} \
        --debug \
        --parallel 0 \
        --chunks 1 \
        --chunk-size 50 \
        --use-pmc \
        --pmc-dof ${PMC_DOF} \
        --pmc-adjust-sample-size 0 \
        --pmc-initialize-from-file ${BASE_NAME}/${scenario}_${data}/mcmc_pre_merged.hdf5 \
        --pmc-hierarchical-clusters ${PMC_CLUSTERS} \
        --global-local-covariance-window ${PMC_PATCH_LENGTH} \
        --global-local-skip-initial ${PMC_SKIP_INITIAL} \
        --pmc-group-by-r-value ${PMC_GROUP_BY_RVALUE} \
        ${PMC_INITIALIZATION} \
        --pmc-draw-samples \
        --pmc-final-chunksize ${PMC_FINAL_CHUNKSIZE} \
        ${!constraints} \
        ${!scan} \
        ${!nuisance} \
        --output "${BASE_NAME}/${scenario}_${data}/hc.hdf5" \
        > ${BASE_NAME}/${scenario}_${data}/hc.log 2>&1
}

export PMC_PARALLEL=1
export PMC_MAX_STEPS=20
export PMC_CHUNKSIZE=3000
export PMC_DOF=-1 ## Gaussian
export PMC_INITIALIZATION="--pmc-patch-around-local-mode 0 --pmc-minimum-overlap 0.05"
export PMC_GROUP_BY_RVALUE=1.5
export PMC_CONVERGENCE="--pmc-relative-std-deviation-over-last-steps 0.2 2 --pmc-ignore-ess 1"
export PMC_CLUSTERS=50
export PMC_FINAL_CHUNKSIZE=1000000
export PMC_PATCH_LENGTH=300
export PMC_SKIP_INITIAL=0.2
export PMC_ADJUST_SAMPLE_SIZE=1
export PMC_IGNORE_GROUPS=""
export PMC_SEED=789654

pmc_monolithic() {
    scenario=${1}
    if [[ -z ${scenario} ]] ; then
        echo "No scenario given!"
        exit -1
    fi
    echo "[scenario = ${scenario}]"

    shift
    data=${1}
    if [[ -z ${data} ]] ; then
        echo "No data given!"
        exit -1
    fi
    echo "[data = ${data}]"

    seed=$PMC_SEED

    scan=SCAN_${scenario}
    constraints=CONSTRAINTS_${data}
    nuisance=NUISANCE_${data}

    mkdir -p ${BASE_NAME}/${scenario}_${data}
    eos-scan-mc \
        --seed ${seed} \
        --debug \
        --parallel ${PMC_PARALLEL} \
        --chunks ${PMC_MAX_STEPS} \
        --chunk-size ${PMC_CHUNKSIZE} \
        --use-pmc \
        --pmc-dof ${PMC_DOF} \
        --pmc-initialize-from-file ${BASE_NAME}/${scenario}_${data}/mcmc_pre_merged.hdf5 \
        --pmc-hierarchical-clusters ${PMC_CLUSTERS} \
        --global-local-covariance-window ${PMC_PATCH_LENGTH} \
        --global-local-skip-initial ${PMC_SKIP_INITIAL} \
        --pmc-final-chunksize ${PMC_FINAL_CHUNKSIZE} \
        --pmc-group-by-r-value ${PMC_GROUP_BY_RVALUE} \
        ${PMC_INITIALIZATION} \
        --pmc-adjust-sample-size ${PMC_ADJUST_SAMPLE_SIZE} \
        ${PMC_CONVERGENCE} \
        ${PMC_IGNORE_GROUPS} \
        ${!constraints} \
        ${!scan} \
        ${!nuisance} \
        --output "${BASE_NAME}/${scenario}_${data}/pmc_monolithic.hdf5" \
        > ${BASE_NAME}/${scenario}_${data}/pmc_monolithic.log 2>&1
}

export PMC_NUMBER_OF_JOBS=200
export PMC_POLLING_INTERVAL=45
export PMC_RESOURCE_MANAGER=${PMC_RESOURCE_MANAGER:-SGE}
export SGE_QUEUE=short
export SGE_FINAL_QUEUE=standard
export SGE_CHECK_ERROR_STATUS=0
export PYTHON=${PYTHON:-python}

pmc_queue() {
    scenario=${1}
    if [[ -z ${scenario} ]] ; then
        echo "No scenario given!"
        exit -1
    fi
    echo "[scenario = ${scenario}]"

    shift
    data=${1}
    if [[ -z ${data} ]] ; then
        echo "No data given!"
        exit -1
    fi
    echo "[data = ${data}]"

    scan=SCAN_${scenario}
    constraints=CONSTRAINTS_${data}
    nuisance=NUISANCE_${data}

    export PMC_ANALYSIS="${!constraints} ${!scan} ${!nuisance}"
    export PMC_OUTPUT_BASE_NAME=${BASE_NAME}/${scenario}_${data}/
    export PMC_MERGE_FILE=$PMC_OUTPUT_BASE_NAME/mcmc_pre_merged.hdf5

    $PYTHON $EOS_SCRIPT_PATH/../jobs/job_manager.py  \
        --resource-manager ${PMC_RESOURCE_MANAGER} \
        $PMC_CLIENT_ARGV \
	2>&1 | tee ${BASE_NAME}/${scenario}_${data}/manager.log
}

gof() {
    scenario=${1}
    if [[ -z ${scenario} ]] ; then
        echo "No scenario given!"
        exit -1
    fi
    echo "[scenario = ${scenario}]"

    shift
    data=${1}
    if [[ -z ${data} ]] ; then
        echo "No data given!"
        exit -1
    fi
    echo "[data = ${data}]"

    shift
    idx=${1}
    if [[ -z ${idx} ]] ; then
        echo "No gof index given!"
        exit -1
    fi

    shift

    # optimize [default] or simply compute gof at given parameter values
    opt=${1}
    if [[ -z ${opt} ]] ; then
        opt=1
    fi

    scan=SCAN_${scenario}
    constraints=CONSTRAINTS_${data}
    nuisance=NUISANCE_${data}
    mode=GOF_MODE_${idx}

    mkdir -p ${BASE_NAME}/${scenario}_${data}

    cmd="--goodness-of-fit"
    if [[ "$opt" -eq 1 ]] ; then
        cmd="${cmd} --optimize"
    fi
    cmd="${cmd} ${!mode}"

    eos-scan-mc \
        --debug \
        ${cmd} \
        ${!constraints} \
        ${!scan} \
        ${!nuisance} \
        --output "${BASE_NAME}/${scenario}_${data}/gof_${idx}.hdf5" \
        > ${BASE_NAME}/${scenario}_${data}/gof_${idx}.log 2>&1
}

export UNC_SAMPLES=100000
export UNC_WORKERS=1 # set equal to number of threads
export UNC_STORE_PAR=1
# "--pmc-input file.hdf INDEX_MIN INDEX_MAX"
export UNC_PMC_INPUT=

unc() {
    scenario=${1}
    if [[ -z ${scenario} ]] ; then
        echo "No scenario given!"
        exit -1
    fi
    echo "[scenario = ${scenario}]"

    shift
    data=${1}
    if [[ -z ${data} ]] ; then
        echo "No set of parameters to vary defined!"
        exit -1
    fi
    echo "[parameters = ${data}]"

    predictions=PREDICTIONS_${data}
    nuisance=NUISANCE_${data}
    nuisance=${!nuisance}
    nuisance=${nuisance//--nuisance/--vary}

    : ${UNC_PARALLEL:=$UNC_WORKERS}

    mkdir -p ${BASE_NAME}/${scenario}_${data}
    eos-propagate-uncertainty \
        --seed $PMC_SEED \
        --samples $UNC_SAMPLES \
        --workers $UNC_WORKERS \
        --parallel $UNC_PARALLEL \
        --store-parameters $UNC_STORE_PAR \
        ${UNC_PMC_INPUT} \
        ${!predictions} \
        ${!nuisance} \
        --output "${BASE_NAME}/${scenario}_${data}/unc.hdf5" \
        > ${BASE_NAME}/${scenario}_${data}/unc_${idx}.log 2>&1
}

unc_queue() {
    scenario=${1}
    if [[ -z ${scenario} ]] ; then
        echo "No scenario given!"
        exit -1
    fi
    echo "[scenario = ${scenario}]"

    shift
    data=${1}
    if [[ -z ${data} ]] ; then
        echo "No set of parameters to vary defined!"
        exit -1
    fi
    echo "[parameters = ${data}]"

    scan=SCAN_${scenario}
    constraints=CONSTRAINTS_${data}
    predictions=PREDICTIONS_${data}
    nuisance=NUISANCE_${data}
    nuisance=${!nuisance}

    export PMC_ANALYSIS="${!constraints} ${!scan} ${!nuisance}"
    export PMC_UNCERTAINTY_ANALYSIS="${!predictions}"
    export PMC_OUTPUT_BASE_NAME=${BASE_NAME}/${scenario}_${data}/

    : ${UNC_PARALLEL:=$UNC_WORKERS}

    mkdir -p ${BASE_NAME}/${scenario}_${data}

    export PMC_INITIALIZATION_MODE="UncertaintyPropagation"

    $PYTHON $EOS_SCRIPT_PATH/../jobs/job_manager.py --uncertainty-propagation \
        --resource-manager ${PMC_RESOURCE_MANAGER} \
        $PMC_CLIENT_ARGV \
	2>&1 | tee ${BASE_NAME}/${scenario}_${data}/manager.log
}

## Job Main Function ##
main() {
    local name=${0}
    if [[ ${name##*/} == slurm_script ]] ; then
        name=${SLURM_JOB_NAME}
    fi

    local name=${name##*/}
    local name=${name%%.*}
    local scenario=${name%%-*}
    local data=${name##*-}

    if [[ -z ${BASE_NAME} ]] ; then
        echo "No BASE_NAME given!"
        exit -1
    fi

    cmd=${1}
    shift

    case ${cmd} in
        pre)
            mcmc_pre ${scenario} ${data} $@
            ;;
        merge)
            mcmc_merge ${scenario} ${data} $@
            ;;
        hc)
            hc ${scenario} ${data} $@
            ;;
        pmc)
            pmc_monolithic ${scenario} ${data} $@
            ;;
        pmc-queue)
            pmc_queue ${scenario} ${data} $@
            ;;
        gof)
            gof ${scenario} ${data} $@
            ;;
        unc)
            unc ${scenario} ${data} $@
            ;;
        unc-queue)
            unc_queue ${scenario} ${data} $@
            ;;
        *)
            echo "No command given!"
            exit -1
            ;;
    esac

}
