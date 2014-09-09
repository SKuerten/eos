#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

export MCMC_PRERUN_SAMPLES=40000
export MCMC_PRERUN_UPDATE_SIZE=500
export MCMC_PRERUN_CHAINS=1
export MCMC_PRERUN_PARALLEL=0

export PMC_ADJUST_SAMPLE_SIZE=1
export PMC_CONVERGENCE="$PMC_CONVERGENCE --pmc-crop-highest-weights 200"
export PMC_CHUNKSIZE=4000
export PMC_CLUSTERS=40
export PMC_DOF=10
export PMC_FINAL_CHUNKSIZE=250000
export PMC_GROUP_BY_RVALUE=2.3
# 0=A,1=C,2=D, 3=B
export PMC_IGNORE_GROUPS="
    --pmc-ignore-group 0
    --pmc-ignore-group 1
    --pmc-ignore-group 2
    --pmc-ignore-group 4
    --pmc-ignore-group 5
    --pmc-ignore-group 6
"
export PMC_PATCH_LENGTH=200
export PMC_SKIP_INITIAL=0.5

export PMC_NUMBER_OF_JOBS=100
export LL_QUEUE=serial
export LL_FINAL_QUEUE=serial
export PMC_POLLING_INTERVAL=60
#export PMC_CLIENT_ARGV="--resume-samples --step 3"

export PMC_UNCERTAINTY_INPUT="${BASE_NAME}/scIII_posthep13/pmc_parameter_samples_15.hdf5_merge"

export GOF_MODE_0="{
 -0.31769256  3.29867058 -4.28361414 -0.0860016  -0.10428679 -0.84031972
  0.82113039  0.22540345  0.13492438  0.36390298  1.27433698  4.19172752
  0.22813541  0.38748441 -4.59415427  0.84061828  1.04400616  0.2824218
  0.15156337  0.27805032 -1.24297324  0.0146963  -0.00946424 -0.0125497
  0.9900188   1.00845228  1.07803377  0.99380949  0.29840775 -2.57132593
 -0.00646443 -0.62632495  0.45419256  0.35039709 }"
export GOF_MODE_1="{
   4.94622633e-01  -4.25128322e+00   4.43393257e+00   8.79652989e-02
  -2.27432836e-01   6.09958103e-01   8.07698846e-01   2.25354235e-01
   1.24821515e-01   3.67671361e-01   1.27558905e+00   4.18684521e+00
   2.27889637e-01   3.68874782e-01  -4.88217405e+00   8.18114332e-01
   1.01092327e+00   2.81092376e-01   1.45935308e-01   2.74727401e-01
  -1.51460641e+00   2.40795152e-03   1.34999183e-03  -5.60503766e-03
   9.59707346e-01   1.01708456e+00   1.10026634e+00   9.50174997e-01
   2.94388912e-01  -2.66401275e+00  -2.73876902e-02   7.36573422e-01
   4.49320644e-01   3.50149251e-01 }"
export GOF_MODE_2="{
   5.82422912e-02   2.94171513e+00  -1.72414796e+00  -4.31590623e-01
   3.35196395e+00   2.77479152e+00   8.01719153e-01   2.25054784e-01
   1.27948100e-01   3.73199312e-01   1.27727889e+00   4.17380983e+00
   2.27054403e-01   3.68183035e-01  -4.86038113e+00   1.01824467e+00
   8.29310889e-01   2.91264302e-01   2.83967307e-01   2.83643710e-01
  -1.75225332e+00  -1.46672907e-03  -1.13914849e-02  -1.22698165e-02
   9.98804171e-01   1.01074788e+00   9.79731256e-01   1.01933202e+00
   3.39217786e-01  -1.78193312e+00  -2.23337239e-02  -9.98646395e-01
   4.48588540e-01   3.49645342e-01 }"
export GOF_MODE_3="{
  0.11607577 -3.80979489  1.74193348  0.4301043  -3.21360002 -2.7154867
  0.79846065  0.22544854  0.10618287  0.35561797  1.26314901  4.17740805
  0.22928935  0.38885097 -5.01989897  1.15069237  0.79472482  0.28333923
  0.11075782  0.27799833 -1.90079342  0.03223771 -0.04027305 -0.0410946
  1.00960507  1.04635071  0.96435021  0.95763938  0.33701753 -1.82198771
 -0.19935003  0.92784446  0.48271053  0.3508929 }"

main $@