source ${EOS_SCRIPT_PATH}/constraints.bash
source ${EOS_SCRIPT_PATH}/predictions.bash
source ${EOS_SCRIPT_PATH}/priors.bash
source ${EOS_SCRIPT_PATH}/scan.bash

mcmc() {
    scenario=${1}
    shift

    data=${1}
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

    echo ${!scan}
    echo ${!constraints}
    echo ${!nuisance}
    echo
    echo "MCMC not implemented!"
    exit -1
}

## Job Main Function ##
main() {
    local name=${0}

    local name=${name##*/}
    local name=${name%%.*}
    local scenario=${name%%-*}
    local data=${name##*-}

    if [[ -z ${BASE_NAME} ]] ; then
        echo "No BASE_NAME given!"
        exit -1
    fi

    if [[ -z ${scenario} ]] ; then
        echo "No scenario given!"
        exit -1
    fi
    echo "[scenario = ${scenario}]"

    if [[ -z ${data} ]] ; then
        echo "No data given!"
        exit -1
    fi
    echo "[data = ${data}]"

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
        gof)
            gof ${scenario} ${data} $@
            ;;
        opt)
            py_opt ${scenario} ${data} $@
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
