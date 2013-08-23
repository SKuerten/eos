#!/bin/bash

scenario=$1; shift
constraints=$1; shift

for i in $(seq -f %03.0f 1 25); do
    file=j${i}.job
    # beware of shell escaping: loadlever variables
    # must not be expanded by this shell
    echo "/bin/bash
#
#@ group =  pr85tu
#@ job_type = serial
#@ class = serial
###                    hh:mm:ss
##@ wall_clock_limit = 20:15:50
#@ job_name = eos-\$(jobid)
#@ initialdir = \$(home)/workspace/eos-scripts/bayes2
#@ output = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).out
#@ error  = /gpfs/work/pr85tu/ru72xaf2/log/\$(jobid).err
#@ notification=always
#@ notify_user=Frederik.Beaujean@lmu.de
#@ queue
. /etc/profile
. /etc/profile.d/modules.sh

export KMP_AFFINITY=granularity=core,compact,1

# output file directory
export BASE_NAME=$BASE_NAME/2013-08-20

./${scenario}-${constraints}.bash pre $i" > $file

    sync
    llsubmit $file
    rm $file
done
