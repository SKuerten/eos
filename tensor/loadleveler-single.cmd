#!/bin/bash

# examples:

# 15 chains
# BASE_NAME=$BASE_NAME/2014-11-28 ./loadleveler-single.cmd sc910TT5 K_KstarBR_Bsmumu mcmc 1 15

scenario=$1; shift
constraints=$1; shift
action=$1; shift
low=$1; shift
high=$1; shift

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

./${scenario}-${constraints}.bash ${action} $i" > $file

    sync
    llsubmit $file
    rm $file
done
