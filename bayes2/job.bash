# vim: set sts=4 et :

## MCMC Prerun ##
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=1000
export MCMC_PRERUN_SCALE_REDUCTION=10
export MCMC_PRERUN_PARALLEL=1

source constraints.bash
source priors.bash
source scan.bash

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
    seed=$((12345 + ${idx} * 1000))

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

export PMC_MONO_PARALLEL=1
export PMC_MONO_CHUNKS=20
export PMC_MONO_CHUNKSIZE=3000
export PMC_MONO_DOF=-1 ## Gaussian
export PMC_MONO_CONVERGENCE="--pmc-relative-std-deviation-over-last-steps 0.05 2 --pmc-ignore-ess 1"
export PMC_MONO_CLUSTERS=50
export PMC_MONO_FINAL_CHUNKSIZE=2000000
export PMC_MONO_COV_WINDOW=300
export PMC_MONO_SKIP_INITIAL=0.2
export PMC_MONO_ADJUST_SAMPLE_SIZE="--pmc-adjust-sample-size"

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

    seed=789654

    scan=SCAN_${scenario}
    constraints=CONSTRAINTS_${data}
    nuisance=NUISANCE_${data}

    mkdir -p ${BASE_NAME}/${scenario}_${data}
    eos-scan-mc \
        --seed ${seed} \
        --debug \
        --parallel ${PMC_MONO_PARALLEL} \
        --chunks ${PMC_MONO_CHUNKS} \
        --chunk-size ${PMC_MONO_CHUNKSIZE} \
        --use-pmc \
        --pmc-dof ${PMC_MONO_DOF} \
        --pmc-initialize-from-file ${BASE_NAME}/${scenario}_${data}/mcmc_pre_merged.hdf5 \
        --pmc-hierarchical-clusters ${PMC_MONO_CLUSTERS} \
        --global-local-covariance-window ${PMC_MONO_COV_WINDOW} \
        --global-local-skip-initial ${PMC_MONO_SKIP_INITIAL} \
        --pmc-final-chunksize ${PMC_MONO_FINAL_CHUNKSIZE} \
        ${PMC_MONO_ADJUST_SAMPLE_SIZE} \
        ${PMC_MONO_CONVERGENCE} \
        ${!constraints} \
        ${!scan} \
        ${!nuisance} \
        --output "${BASE_NAME}/${scenario}_${data}/pmc_monolithic.hdf5" \
        > ${BASE_NAME}/${scenario}_${data}/pmc_monolithic.log 2>&1
}


## Job Main Function ##
main() {
    local name=${0##*/}
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
        pmc)
            pmc_monolithic ${scenario} ${data} $@
            ;;
        *)
            echo "No command given!"
            exit -1
            ;;
    esac
}
