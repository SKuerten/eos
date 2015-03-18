# defines the allowed parameter range in terms of sigmas for Gaussian and LogGamma distributions
export N_SIGMAS=3

# From UTfit Summer 2013 (post-EPS13)
# Results of the Treelevel Fit
export NUISANCE_CKM_posthep13="
    --nuisance    CKM::A                                                    ${N_SIGMAS}     --prior    gaussian    0.786     0.806     0.826
    --nuisance    CKM::lambda                                               ${N_SIGMAS}     --prior    gaussian    0.2247    0.2253    0.2259
    --nuisance    CKM::rhobar                                    0.0   1.0  ${N_SIGMAS}     --prior    gaussian    0.083     0.132     0.181
    --nuisance    CKM::etabar                                               ${N_SIGMAS}     --prior    gaussian    0.319     0.369     0.419
"

# From PDG 2012
export NUISANCE_QUARK_MASSES_PDG2012="
    --nuisance    mass::c                                        0.0   2.0  ${N_SIGMAS}     --prior    gaussian    1.25      1.275     1.30
    --nuisance    mass::b(MSbar)                                            ${N_SIGMAS}     --prior    gaussian    4.15      4.18      4.21
"

export NUISANCE_DECAY_CONSTANTS_flag13="
    --nuisance    decay-constant::B_s                                       ${N_SIGMAS}     --prior    gaussian    0.2232    0.2277    0.2322
"

#####################
## form factors    ##
#####################
export NUISANCE_B_TO_VPERPLL_FORM_FACTORS_KMPW2010="
    --global-option form-factors KMPW2010

    --nuisance    B->K^*::F^V(0)@KMPW2010                                   ${N_SIGMAS}     --prior    gaussian   0.24      0.36      0.59
    --nuisance    B->K^*::b^V_1@KMPW2010                        -6.0   5.4  ${N_SIGMAS}     --prior    gaussian  -5.20     -4.80     -4.00
"

export NUISANCE_B_TO_VPARALL_FORM_FACTORS_KMPW2010="
    --global-option form-factors KMPW2010

    --nuisance    B->K^*::F^A1(0)@KMPW2010                                  ${N_SIGMAS}     --prior    gaussian   0.15      0.25      0.41
    --nuisance    B->K^*::b^A1_1@KMPW2010                       -2.06  5.4  ${N_SIGMAS}     --prior    gaussian  -0.46     +0.34     +1.20
    --nuisance    B->K^*::F^A2(0)@KMPW2010                                  ${N_SIGMAS}     --prior    gaussian   0.13      0.23      0.42
    --nuisance    B->K^*::b^A2_1@KMPW2010                       -4.9   5.4  ${N_SIGMAS}     --prior    gaussian  -2.20     -0.85     +2.03
"

# naive average of [BZ2004] and [KMPW2010] for the mean values. Widen
# uncertainty to provide larger tail on the short side
# b^p taken from [KMPW2010], not given in [BZ2004]
export NUISANCE_B_TO_PLL_FORM_FACTORS_PLUS_KMPW2010="
    --global-option form-factors KMPW2010

    --nuisance    B->K::F^p(0)@KMPW2010                          0.19  0.49 ${N_SIGMAS}     --prior    gaussian   0.29      0.34      0.39
    --nuisance    B->K::b^p_1@KMPW2010                          -6.9   0.6  ${N_SIGMAS}     --prior    gaussian  -3.7      -2.1      -1.2
"

# F^0(0)=F^p(0) => eliminate F^0(0)
export NUISANCE_B_TO_PLL_FORM_FACTORS_SCALAR_KMPW2010="
    --global-option form-factors KMPW2010

    --nuisance    B->K::b^0_1@KMPW2010                          -7.0   1.9  ${N_SIGMAS}     --prior    gaussian  -5.2      -4.3      -3.5
"

# see F^p
export NUISANCE_B_TO_PLL_FORM_FACTORS_TENSOR_KMPW2010="
    --global-option form-factors KMPW2010

    --nuisance    B->K::F^t(0)@KMPW2010                          0.20  0.56 ${N_SIGMAS}     --prior    gaussian   0.32      0.38      0.44
    --nuisance    B->K::b^t_1@KMPW2010                          -8.2   0.8  ${N_SIGMAS}     --prior    gaussian  -4.2      -2.2      -1.2
"

export NUISANCE_B_TO_PLL_FORM_FACTORS_ALL_KMPW2010="
    ${NUISANCE_B_TO_PLL_FORM_FACTORS_PLUS_KMPW2010}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS_SCALAR_KMPW2010}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS_TENSOR_KMPW2010}
"

#####################
## subleading      ##
#####################

export NUISANCE_B_TO_VPERPLL_SUBLEADING="
    --nuisance    B->Vll::Lambda_pp@LowRecoil                   -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->K^*ll::A_perp^L_uncertainty@LargeRecoil     0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_perp^R_uncertainty@LargeRecoil     0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
"

export NUISANCE_B_TO_VPARALL_SUBLEADING="
    --nuisance    B->Vll::Lambda_0@LowRecoil                    -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->Vll::Lambda_pa@LowRecoil                   -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->K^*ll::A_0^L_uncertainty@LargeRecoil        0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_0^R_uncertainty@LargeRecoil        0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_par^L_uncertainty@LargeRecoil      0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_par^R_uncertainty@LargeRecoil      0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
"

export NUISANCE_B_TO_PLL_SUBLEADING="
    --nuisance    B->Pll::Lambda_pseudo@LowRecoil               -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->Pll::Lambda_pseudo@LargeRecoil             -1.0   1.0  ${N_SIGMAS}     --prior    gaussian   -0.50      0.0       0.50
"

#####################
## named parameter sets
#####################

export NUISANCE_FH="
$NUISANCE_CKM_posthep13
$NUISANCE_QUARK_MASSES_PDG2012
$NUISANCE_B_TO_PLL_FORM_FACTORS_ALL_KMPW2010
$NUISANCE_B_TO_PLL_SUBLEADING
"

export NUISANCE_K_KstarBR="
$NUISANCE_FH
$NUISANCE_B_TO_VPARALL_SUBLEADING
$NUISANCE_B_TO_VPERPLL_SUBLEADING
$NUISANCE_B_TO_VPERPLL_FORM_FACTORS_KMPW2010
$NUISANCE_B_TO_VPARALL_FORM_FACTORS_KMPW2010
"

export NUISANCE_K_KstarBR_no_ACP=$NUISANCE_K_KstarBR

export NUISANCE_K_KstarBR_Bsmumu="
$NUISANCE_K_KstarBR
$NUISANCE_DECAY_CONSTANTS_flag13
"

export NUISANCE_K_KstarBR_no_ACP_Bsmumu=$NUISANCE_K_KstarBR_Bsmumu

export NUISANCE_Bsmumu="
$NUISANCE_CKM_posthep13
$NUISANCE_DECAY_CONSTANTS_flag13
"

export NUISANCE_FH_Bsmumu="
$NUISANCE_CKM_posthep13
$NUISANCE_QUARK_MASSES_PDG2012
$NUISANCE_B_TO_PLL_FORM_FACTORS_ALL_KMPW2010
$NUISANCE_B_TO_PLL_SUBLEADING
$NUISANCE_DECAY_CONSTANTS_flag13
"
export NUISANCE_test1=$NUISANCE_K_KstarBR_Bsmumu
export NUISANCE_test2=$NUISANCE_test1
export NUISANCE_test3=$NUISANCE_test1
export NUISANCE_test4=$NUISANCE_test1
