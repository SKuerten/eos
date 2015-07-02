#!/bin/bash

# Uncertainty propagation on a set of input samples with trivial
# parallelization.
#
# usage example:
scenario=$1; shift
observable=$1; shift
input=$1; shift
njobs=$1; shift

name=${scenario}_${observable}

file=$name.job

# beware of shell escaping: loadlever variables
# must not be expanded by this shell
echo "#! /bin/bash
#@ job_name = unc_$name
#@ group = pr85tu
#@ job_type = serial
#@ class = serial
#@ node_usage = shared
#@ resources = ConsumableCpus(1)
#
###                    hh:mm:ss
#@ wall_clock_limit = 15:59:50
#@ initialdir = \$(home)/workspace/eos-scripts/tensor
#@ output = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).out
#@ error  = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).err
#@ notification=error
#@ notify_user=Frederik.Beaujean@lmu.de
#@ environment = \$BASE_NAME
#@ executable = ./unc_job.bash
" > $file

if [ ! -f $input ]; then
    echo "Input file $input does not exist"
    exit -1
fi

###
# extract the number of samples
###
# ack-grep not available everywhere but so much easier to read!
# nsamples=$(h5ls -r $input| ack '/chain\\ #0/samples' | ack -ho "Dataset {(\d*)," --output '$1')

# egrep -o only output the match instead of the full line
nsamples=$(h5ls -r $input | grep '/chain\\ #0/samples' | egrep -o '\{.*,')
# remove '{' and ',' from '{20000,'
nsamples=${nsamples:1:${#nsamples}-2}
if [ -z $nsamples ]; then
    # try with importance samples

    # s=/step\ #5/combination/weights\ #5 Dataset {99968, 1}
    s=$(h5ls -r $input | egrep 'step.*/combination/weights' | tail -n1)
    # $3 = "#5"
    nsteps=$(echo "$s" | awk -F'[ ,]+' '{print $3}' | cut -c 2-)
    # $5 ="{99968"
    samples=$(echo "$s" | awk -F'[ ,]+' '{print $5}' | cut -c 2-)
    # start counting at zero => + 1
    nsamples=$(( (nsteps + 1) * samples))
fi
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
#@ arguments = $scenario $observable $input $low $high
#@ queue
" >> $file
done

# cat $file
sync
llsubmit $file
rm $file
