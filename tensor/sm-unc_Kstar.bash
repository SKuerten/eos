#!/bin/bash

source ${EOS_SCRIPT_PATH}/job.bash

export EOS_SEED=74124

export EOS_ANALYSIS_INFO=0

# LCSR + lattice
export GOF_MODE_0="{ +0.8067709803581238 +0.2250771075487137 +0.1742186844348907 +0.3394549787044525 +1.2889647483825684 +4.1742711067199707 +0.0322905741631985 +0.1163743883371353 +0.9008494615554810 +0.9220937490463257 +1.0581090450286865 +1.0897251367568970 +0.0449241101741791 +1.0402346849441528 +1.0500251054763794 +0.3437604308128357 -1.2188867330551147 +1.0193508863449097 +0.2589139044284821 +0.0377540588378906 -0.0092262811958790 +0.4762068390846252 +0.2106992602348328 +0.3468116521835327 -1.2326784133911133 +1.6714459657669067 +0.3041836023330688 -1.1160444021224976 +1.6197041273117065 +0.2673358917236328 +0.0841161012649536 +0.6525657176971436 +0.8097377419471741 -0.4486638009548187 }"
# BOBYQA output from mcmc input above
export GOF_MODE_0="{
 +0.8059999918438847 +0.2253000226921414 +0.1319911845876919
 +0.3690032258750632 +1.2749999080153291 +4.1800010258047227
 +0.0000091067424079 +0.0000193011574175 +0.9999808157951967
 +1.0000056404342050 +1.0000312264896634 +0.9999803896932090
 -0.0000196389707319 +1.0000021865613131 +0.9999963270114348
 +0.3516853088428650 -1.2076067448640151 +0.7609658923394123
 +0.2626435469688269 +0.0625696094378624 +0.0545783712314495
 +0.5255114358961852 +0.2934663918032049 +0.3452748355822677
 -1.1869943928412585 +1.5699695542958305 +0.3107796514953096
 -1.0682361142306289 +1.4072953757822966 +0.3307307884971034
 +0.2603637837810474 +0.6660234960295126 +0.9637539437640690
 -0.0635403125267390 }"

# only LCSR
export GOF_MODE_1="{
 +0.8060036474243922 +0.2252999274976096 +0.1320007567807374
 +0.3689980246083343 +1.2750023515207443 +4.1800052681487045
 +0.0000018713920043 +0.0000184886183209 +0.9999482051924227
 +0.9999776719347816 +0.9999959630416734 +1.0000268005141020
 +0.0000067183229778 +0.9999704296343943 +1.0000107588140512
 +0.3620821702107370 -1.1869904585131599 +0.5359785555728591
 +0.2675907858702194 +0.0817140985638702 +0.0510467037559331
 +0.5840461227250386 +0.4653974333961544 +0.3451924054329074
 -1.1735951779577452 +1.4154176732298789 +0.3126634610719649
 -1.0527969553225902 +1.2751356109642780 +0.3520934415418787
 +0.3120504276302807 +0.7315109982347804 +1.3448435272840336
 +0.6563929773901370 }"

export EOS_IS_SAMPLES=50000

export EOS_MCMC_ACCEPTANCE_MAX=0.35
export EOS_MCMC_BURN_IN=
# export EOS_MCMC_COVARIANCE="Kstar-FF-cov.txt"
# export EOS_MCMC_INITIAL_VALUES="fixed"
export EOS_MCMC_INTEGRATION_POINTS=16
export EOS_MCMC_PROPOSAL='gauss'
export EOS_MCMC_SAMPLES=50000
# export EOS_MCMC_SCALE_NUISANCE=
# export EOS_MCMC_SCALE_REDUCTION=1
export EOS_MCMC_UPDATE_SIZE=500

export EOS_OPT_MAXEVAL=50000

export EOS_VB_COMPONENTS_PER_GROUP=15
export EOS_VB_EXTRA_OPTIONS=
export EOS_VB_INIT_METHOD="random"
export EOS_VB_PRUNE=
export EOS_VB_MCMC_INPUT=
export EOS_VB_SKIP_INITIAL=0
export EOS_VB_THIN=1
export EOS_VB_REL_TOL=
export EOS_VB_R_VALUE=2


main $@
