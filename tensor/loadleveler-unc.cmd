#!/bin/bash

# Uncertainty propagation on a set of input samples with trivial
# parallelization.
#
# usage example:

script=./$1; shift
njobs=$1; shift

# file name without path and extension, example:
# ./${scenario}-${constraints}.bash
script_name=${script##*/}
script_name=${script_name%%.*}
if [ ! -f $script ]; then
    echo "Script $script not found!"
    exit -1
fi

file=${script_name}.job

# beware of shell escaping: loadlever variables
# must not be expanded by this shell
echo "#! /bin/bash
#@ job_name = unc_${script_name}
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

if [ ! -f $EOS_UNC_INPUT ]; then
    echo "Input file ${EOS_UNC_INPUT} does not exist"
    exit -1
fi

###
# extract the number of samples
###
# ack-grep not available everywhere but so much easier to read!
# nsamples=$(h5ls -r $EOS_UNC_INPUT| ack '/chain\\ #0/samples' | ack -ho "Dataset {(\d*)," --output '$1')

# egrep -o only output the match instead of the full line
nsamples=$(h5ls -r $EOS_UNC_INPUT | grep '/chain\\ #0/samples' | egrep -o '\{.*,')
# remove '{' and ',' from '{20000,'
nsamples=${nsamples:1:${#nsamples}-2}
if [ -z $nsamples ]; then
    echo "Could not determine number of input samples"
    exit -1
fi

# number of samples per job, integer division!
nperjob=$(($nsamples / $njobs))

for i in $(seq -f %01.0f 1 $njobs ); do
    low=$(((i-1) * nperjob))
    if [ $i -lt $njobs ]; then
        high=$((i * nperjob))
    else
        # last job gets rest of samples if nsamples not divisible by njobs
        high=$nsamples
    fi
    echo "
#@ step_name = step_${i}_${low}_${high}
#@ arguments = unc $low $high
#@ queue
" >> $file
done

# cat $file
sync
llsubmit $file
rm $file
