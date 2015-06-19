#!/bin/bash

# examples:

# 15 chains
# BASE_NAME=$WORK/eos/2015-tensor/2015-03-09 ./loadleveler-single.cmd sm-test1.bash mcmc 1 30

script=./$1; shift
action=$1; shift
low=$1; shift
high=$1; shift

# file name without path and extension, example:
# ./${scenario}-${constraints}.bash
name=${script##*/}
name=${name%%.*}
if [ ! -f $script ]; then
    echo "Script $script not found!"
    exit -1
fi

file=${name}.job

# beware of shell escaping: loadlever variables
# must not be expanded by this shell
echo "#! /bin/bash
#@ job_name = ${name}
#@ group = pr85tu
#@ job_type = serial
#@ class = serial
#@ node_usage = shared
#@ resources = ConsumableCpus(1)
#
###                    hh:mm:ss
#@ wall_clock_limit = 47:59:55
#@ initialdir = \$(home)/workspace/eos-scripts/tensor
#@ output = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).out
#@ error  = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).err
#@ notification=error
#@ notify_user=Frederik.Beaujean@lmu.de
#@ environment = \$BASE_NAME
#@ executable = $script
" > $file

# all jobs are independent
for i in $(seq -f %01.0f $low $high); do
   echo "
#@ step_name = ${action}_$i
#@ arguments = $action $i
#@ queue
" >> $file
done

sync
llsubmit $file
rm $file
