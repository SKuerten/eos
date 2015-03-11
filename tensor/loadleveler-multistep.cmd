#!/bin/bash

# examples:

# importance sampling at step 1 with 200 processes
# BASE_NAME=$BASE_NAME/2014-11-28 ./loadleveler-single.cmd sc910TT5 K_KstarBR_Bsmumu is 1 200

scenario=$1; shift
constraints=$1; shift
steps=$1; shift
nnode=$1; shift

script=./${scenario}-${constraints}.bash
if [ ! -f $script ]; then
   echo "Script $script not found!"
   exit -1
fi
# job class
# job_class=shorttest

# job file
file=${scenario}_${constraints}_${steps}_${nnode}.job

common() {
echo "#! /bin/bash
#@ job_name = tensor-\$(jobid)
#@ group = pr85tu
#@ resources = ConsumableCpus(1)
###                   hh:mm:ss
#@ wall_clock_limit = 19:59:50
#@ initialdir = \$(home)/workspace/eos-scripts/tensor
#@ output = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).out
#@ error  = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).err
#@ notification=error
#@ notify_user=Frederik.Beaujean@lmu.de
"
}

sampling_step() {
step=$1; shift

# beware of shell escaping: loadlever variables
# must not be expanded by this shell
echo "
#@ step_name = is_${step}_\$(jobid)
#@ job_type = parallel
#@ class = shorttest
#
# shared nodes
#
##@ node_usage = shared
##@ total_tasks = $nnode
##@ blocking = unlimited
#
# full nodes
#
#@ node_usage = not_shared
#@ node = $nnode
#@ tasks_per_node = 32
#
#@ executable = /usr/bin/poe
#@ arguments = $script is $step
#@ queue
"
}

vb_step() {
step=$1; shift

echo "
#@ step_name = vb_${step}_\$(jobid)
#@ job_type = serial
#@ class = shorttest
#@ executable = /usr/bin/poe
#@ arguments = $script vb is $step APPEND
#@ queue
"
}

# output file directory
export BASE_NAME=$BASE_NAME

export OMP_NUM_THREADS=1
export MP_TASK_AFFINITY=cpu:$OMP_NUM_THREADS

common >> $file

# number of VB steps
let n_vb=$steps-1
let i=0

for ((; i < n_vb; i++)); do
    sampling_step $i >> $file
    vb_step $i >> $file
done

sampling_step $i >> $file

#cat $file
sync
llsubmit $file
rm $file
