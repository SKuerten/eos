#!/bin/bash

# usage example: run 5 steps on one node
# BASE_NAME=$WORK/eos/2015-tensor/2015-03-09 ./loadleveler-multistep.cmd scSP-Bsmumu.bash 5 1

script=./$1; shift
steps=$1; shift
nnode=$1; shift

# file name without path and extension, example:
# ./${scenario}-${constraints}.bash
name=${script##*/}
name=${name%%.*}
if [ ! -f $script ]; then
   echo "Script $script not found!"
   exit -1
fi
# job file
file=${name}_${steps}_${nnode}.job

export OMP_NUM_THREADS=1
export MP_TASK_AFFINITY=cpu:$OMP_NUM_THREADS
# create stack trace if MPI timeout occurs
export MP_DEBUG_TIMEOUT_COMMAND=~/.local/bin/timeout_debug.sh

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
## any environment variable set in .profile overwrites the below settings!
#@ environment = \$BASE_NAME; \$OMP_NUM_THREADS; \$MP_TASK_AFFINITY;
"
}

sampling_step() {
let step=$1; shift
if [[ $step -gt 0 ]]; then
    dependency="#@ dependency = vb_${step} == 0"
fi

tasks_per_node=16

# beware of shell escaping: loadlever variables
# must not be expanded by this shell
echo "
#@ step_name = is_${step}
$dependency
#@ job_type = mpich
#@ class = parallel
#
#@ node_usage = not_shared
#@ node = $nnode
#@ tasks_per_node = ${tasks_per_node}
#
#@ executable = $(which mpirun)
#@ arguments = -n $((tasks_per_node * nnode)) $script is $step
#@ queue
"
}

vb_step() {
step=$1; shift
let isstep=$step-1
echo "
#@ step_name = vb_${step}
#@ dependency = is_${isstep} == 0
#@ job_type = serial
#@ class = serial
#@ executable = $script
#@ arguments = vb is $step APPEND
#@ queue
"
}

create_job_file() {
    common >> $file

    sampling_step 0 >> $file
    for ((i=1; i <= steps; i++)); do
        vb_step $i >> $file
        sampling_step $i >> $file
    done
}

create_job_file
# cat $file
sync
llsubmit $file
rm $file
