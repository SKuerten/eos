#!/bin/bash

# Uncertainty propagation on a set of input samples with trivial
# parallelization.
#
# usage example:

script=./$1; shift
njobs=$1; shift

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
#@ job_name = unc_${name}
#@ group = pr85tu
#@ job_type = serial
#@ class = serial
#@ node_usage = shared
#@ resources = ConsumableCpus(1)
#
###                    hh:mm:ss
#@ wall_clock_limit = 47:59:50
#@ initialdir = \$(home)/workspace/eos-scripts/tensor
#@ output = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).out
#@ error  = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).err
#@ notification=error
#@ notify_user=Frederik.Beaujean@lmu.de
#@ environment = \$BASE_NAME
#@ executable = $script
" > $file

# script expects action. A little hack to do nothing,
# we just want the variables defined in the script.
source $script noop

# extract the number of samples
let nsamples=`h5ls -r $EOS_UNC_INPUT | grep ...`
echo $EOS_UNC_INPUT
# all jobs are independent
for i in $(seq -f %01.0f 1 $njobs); do
   echo "
#@ step_name = $i
#@ arguments = $i
#@ queue
" >> $file
done

cat $file
# sync
# llsubmit $file
# rm $file
