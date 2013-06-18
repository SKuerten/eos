#! /bin/bash

# Time and execute a single command
#
# $1 - The command
# $2 - If '-p', only print command, w/o actually executing it
time_and_run()
{
    local com=$1; shift
    local arg=$1; shift

    echo $com
    # don't execute command
    if [ "$arg" == '-p' ]; then
        return 0
    fi
    eval time $com
}

# Perform prerun
#
# $1 - If '-p', only print commands, w/o actually executing them
preruns()
{
    local arg=$1; shift

    local scenarios=( BPll_largeRec BPll_lowRec ) # all_nuis all_wide BVll_largeRec )
    local n_jobs=30

    for scen in "${scenarios[@]}" ; do
        time_and_run "qsub -t 1-$n_jobs -N $scen ./1-real/sge.sh $scen --pre" $arg
    done
}

#  Determine GOF
gof()
{
    local arg=$1; shift

    # declare -A n_modes
    # n_modes[all_nuis] = 5
    # n_modes[BPll] = 6
    # n_modes[BVll_largeRec] = 6
    # n_modes[BVll_lowRec] = 4

    # hand made associative array; bash 3.2 doesn't have it yet.
    local n_modes=( 6 6 6 4 1 )
    local scenarios=( all_nuis BPll BVll_largeRec BVll_lowRec SM_all )

    
    # loop over array indices with i -> ${!n_modes[*]}
    for i in 3 ; do
        time_and_run "qsub -t 1-${n_modes[$i]} -N ${scenarios[$i]} ./1-real/sge.sh ${scenarios[$i]} --gof" $arg            
    done
}

# Run an action on all subscenarios, one after the other
#
# $1 - If '-p', only print commands, w/o actually executing them
pmc()
{
    local arg=$1; shift

    local script='./1-real/Scenario1.job'
    local action='--pmc-queue'
    #local action='--hc'

    local scenarios=( BPll_largeRec BPll_lowRec ) # BPll BVll_largeRec BVll_lowRec     all_wide all_nuis

    for scen in "${scenarios[@]}" ; do
        time_and_run "$script $scen $action" $arg
    done
}

main()
{
    local arg=$1; shift

#    pmc $arg
     preruns $arg
#     gof $arg
}
main "$@"
