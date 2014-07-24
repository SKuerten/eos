#!/bin/bash

#
#@ group =  pr85tu
#@ job_type = parallel
#@ class = parallel
###                    hh:mm:ss
##@ wall_clock_limit = 40:55:50
#@ node = 1
#@ total_tasks = 1
#@ node_usage = shared
#@ resources = ConsumableCpus(12)
#
#@ job_name = eos-$(jobid)
#@ initialdir = $(home)/workspace/eos-scripts/bayes2
#@ output = /gpfs/work/pr85tu/ru72xaf2/log/$(jobid).out
#@ error  = /gpfs/work/pr85tu/ru72xaf2/log/$(jobid).err
#@ notification=error
#@ notify_user=Frederik.Beaujean@lmu.de
#@ queue
. /etc/profile
. /etc/profile.d/modules.sh

export KMP_AFFINITY=granularity=core,compact,1

# output file directory
export BASE_NAME=$BASE_NAME/2013-08-28

./sm-quim1.bash pmc
