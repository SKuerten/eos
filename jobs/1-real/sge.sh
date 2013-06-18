#! /bin/bash
#$ -q short
#$ -j y
#$ -M beaujean@mpp.mpg.de
#$ -m eas
#$ -o /afs/ipp-garching.mpg.de/home/f/fdb/JobOutput/Scenario1.log
#$ -V
#$ -cwd
#$ -S /bin/bash

# start counting at 0, not 1
i=$(expr $SGE_TASK_ID - 1)

# extract args
scen=$1; shift
action=$1; shift

$HOME/workspace/Sandbox/eos/jobs/1-real/Scenario1.job $scen $action $i
