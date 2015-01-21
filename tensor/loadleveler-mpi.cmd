#!/bin/bash

# examples:

# importance sampling at step 1 with 200 processes
# BASE_NAME=$BASE_NAME/2014-11-28 ./loadleveler-single.cmd sc910TT5 K_KstarBR_Bsmumu is 1 200

scenario=$1; shift
constraints=$1; shift
action=$1; shift
step=$1; shift
nproc=$1; shift

file=${scenario}_${constraints}_${action}_${step}_${nproc}.job

# beware of shell escaping: loadlever variables
# must not be expanded by this shell
echo "#! /bin/bash
#
#@ group = pr85tu
#@ job_type = parallel
#@ class = parallel
#@ node_usage = shared
#@ blocking = unlimited
#@ resources = ConsumableCpus(1)
#@ total_tasks = $nproc
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

export MP_TASK_AFFINITY=cpu:1

# output file directory
export BASE_NAME=$BASE_NAME

poe ./${scenario}-${constraints}.bash ${action} ${step}" > $file

sync
llsubmit $file
cat $file
rm $file
