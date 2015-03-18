#!/bin/bash

# examples:

# 15 chains
# BASE_NAME=$WORK/eos/2015-tensor/2015-03-09 ./loadleveler-single.cmd sm-test1.bash mcmc 1 30

script=./$1; shift
action=$1; shift
low=$1; shift
high=$1; shift

if [ ! -f $script ]; then
    echo "Script $script not found!"
    exit -1
fi

for i in $(seq -f %01.0f $low $high); do
    file=j${i}.job
    # beware of shell escaping: loadlever variables
    # must not be expanded by this shell
    echo "#! /bin/bash
#
#@ group = pr85tu
#@ job_type = serial
#@ class = serial
#@ node_usage = shared
#@ resources = ConsumableCpus(1)
#
###                    hh:mm:ss
#@ wall_clock_limit = 47:59:50
#@ job_name = eos-\$(jobid)
#@ initialdir = \$(home)/workspace/eos-scripts/tensor
#@ output = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).out
#@ error  = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).err
#@ notification=error
#@ notify_user=Frederik.Beaujean@lmu.de
#@ queue

# output file directory
export BASE_NAME=$BASE_NAME

${script} ${action} $i" > $file

    sync
    llsubmit $file
    rm $file
done
