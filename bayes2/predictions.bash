# vim: set sts=4 et :

echo "[predictions loaded]"

#####################
## Uncertainty propagation
#####################

KSTAR0_MU_MU_OPT=",q=d,l=mu,form-factors=KMPW2010,model=WilsonScan"

# Return string def of large recoil observables for uncertainty propagation
# expect two parameters: s_min, s_max
bvll_large_recoil()
{
 observables="
    --kinematics s_min  BIN_MIN  --kinematics s_max  BIN_MAX  --observable B->K^*ll::P'_4@LargeRecoil${KSTAR0_MU_MU_OPT}
    --kinematics s_min  BIN_MIN  --kinematics s_max  BIN_MAX  --observable B->K^*ll::P'_5@LargeRecoil${KSTAR0_MU_MU_OPT}
    --kinematics s_min  BIN_MIN  --kinematics s_max  BIN_MAX  --observable B->K^*ll::P'_6@LargeRecoil${KSTAR0_MU_MU_OPT}
"
observables=${observables//BIN_MIN/$1}
observables=${observables//BIN_MAX/$2}

echo $observables
}

bvll_low_recoil()
{
 observables="
    --kinematics s_min  BIN_MIN  --kinematics s_max  BIN_MAX  --observable B->K^*ll::P'_4@LowRecoil${KSTAR0_MU_MU_OPT}
    --kinematics s_min  BIN_MIN  --kinematics s_max  BIN_MAX  --observable B->K^*ll::P'_5@LowRecoil${KSTAR0_MU_MU_OPT}
    --kinematics s_min  BIN_MIN  --kinematics s_max  BIN_MAX  --observable B->K^*ll::P'_6@LowRecoil${KSTAR0_MU_MU_OPT}
"
observables=${observables//BIN_MIN/$1}
observables=${observables//BIN_MAX/$2}

echo $observables
}

# stop at q^2=19 because of LHCb arXiv:1307.1707
export PREDICTIONS_uncVLL="
    $(bvll_large_recoil 2 4.3)
    $(bvll_large_recoil 1 6)
    $(bvll_low_recoil 14.18 16)
    $(bvll_low_recoil 16 19.00)
"
export PREDICTIONS_uncVLLwide=${PREDICTIONS_uncVLL}

export PREDICTIONS_uncINCL="
    --kinematics E_min 1.8 --observable B->X_sgamma::BR(E_min)@NLO,model=WilsonScan
    --kinematics s_min 1 --kinematics s_max 6 --observable B->X_sll::BR@HLMW2005,model=WilsonScan
"

export PREDICTIONS_uncFF="
    --kinematics s 0.0 --observable B->K^*::V(s)/A_1(s),form-factors=KMPW2010
    --kinematics s 0.0 --observable B->K^*::A_2(s)/A_1(s),form-factors=KMPW2010
"
