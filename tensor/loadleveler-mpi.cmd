#!/bin/bash

# examples:

# importance sampling at step 1 with 200 processes
# BASE_NAME=$BASE_NAME/2014-11-28 ./loadleveler-single.cmd sc910TT5 K_KstarBR_Bsmumu is 1 200

scenario=$1; shift
constraints=$1; shift
action=$1; shift
step=$1; shift
n=$1; shift

script=./${scenario}-${constraints}.bash
if [ ! -f $script ]; then
    echo "Script $script not found!"
    exit -1
fi

file=${scenario}_${constraints}_${action}_${step}_${n}.job

# beware of shell escaping: loadlever variables
# must not be expanded by this shell
echo "#! /bin/bash
#
#@ group = pr85tu
#@ job_type = parallel
#@ class = shorttest
#@ resources = ConsumableCpus(1)
#
# shared nodes
#
##@ node_usage = shared
##@ total_tasks = $n
##@ blocking = unlimited
#
# full nodes
#
#@ node_usage = not_shared
#@ node = $n
#@ tasks_per_node = 32
##@ network.MPI = sn_all,not_shared,us
#
###                   hh:mm:ss
#@ wall_clock_limit = 19:59:50
#@ job_name = tensor-\$(jobid)
#@ initialdir = \$(home)/workspace/eos-scripts/tensor
#@ output = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).out
#@ error  = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).err
#@ notification=error
#@ notify_user=Frederik.Beaujean@lmu.de
#@ queue

# output file directory
export BASE_NAME=$BASE_NAME

export OMP_NUM_THREADS=1
export MP_TASK_AFFINITY=cpu:$OMP_NUM_THREADS

poe ./${script}.bash ${action} ${step}" > $file

sync
llsubmit $file
#cat $file
rm $file
