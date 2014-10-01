# vim: set sts=4 et :

echo "[priors loaded]"

# defines the allowed parameter range in terms of sigmas for Gaussian and LogGamma distributions
export N_SIGMAS=3

# From UTfit Winter 2013 (pre-Moriond13)
# Results of the Treelevel Fit
export NUISANCE_CKM="
    --nuisance    CKM::A                                                    ${N_SIGMAS}     --prior    gaussian    0.787     0.807     0.827
    --nuisance    CKM::lambda                                               ${N_SIGMAS}     --prior    gaussian    0.22470   0.22535   0.22600
    --nuisance    CKM::rhobar                                    0.0   1.0  ${N_SIGMAS}     --prior    gaussian    0.073     0.128     0.183
    --nuisance    CKM::etabar                                               ${N_SIGMAS}     --prior    gaussian    0.315     0.375     0.435
"

# From UTfit Summer 2013 (post-EPS13)
# Results of the Treelevel Fit
export NUISANCE_CKM_posthep13="
    --nuisance    CKM::A                                                    ${N_SIGMAS}     --prior    gaussian    0.786     0.806     0.826
    --nuisance    CKM::lambda                                               ${N_SIGMAS}     --prior    gaussian    0.2247    0.2253    0.2259
    --nuisance    CKM::rhobar                                    0.0   1.0  ${N_SIGMAS}     --prior    gaussian    0.083     0.132     0.181
    --nuisance    CKM::etabar                                               ${N_SIGMAS}     --prior    gaussian    0.319     0.369     0.419
"

# From PDG 2012
export NUISANCE_QUARK_MASSES="
    --nuisance    mass::c                                        0.0   2.0  ${N_SIGMAS}     --prior    gaussian    1.25      1.275     1.30
    --nuisance    mass::b(MSbar)                                            ${N_SIGMAS}     --prior    gaussian    4.15      4.18      4.21
"

# From Lattice Averages, June 2013
export NUISANCE_DECAY_CONSTANTS="
    --nuisance    decay-constant::B_s                                       ${N_SIGMAS}     --prior    gaussian    0.2226    0.2276    0.2326
"
#    --nuisance    decay-constant::B_d                                       ${N_SIGMAS}     --prior    gaussian    0.1859    0.1906    0.1953
#    --nuisance    decay-constant::K_d                                       ${N_SIGMAS}     --prior    gaussian    0.1549    0.1561    0.1573


export NUISANCE_DECAY_CONSTANTS_flag13="
    --nuisance    decay-constant::B_s                                       ${N_SIGMAS}     --prior    gaussian    0.2232    0.2277    0.2322
"

#export NUISANCE_B_TO_VLL_HADRONICS="
#    --nuisance    B->K^*::f_Kstar_perp@2GeV                                 ${N_SIGMAS}     --prior    gaussian    0.168     0.173     0.178
#    --nuisance    B->K^*::f_Kstar_par                                       ${N_SIGMAS}     --prior    gaussian    0.212     0.217     0.222
#    --nuisance    lambda_B_p                                     0.1   0.9  ${N_SIGMAS}     --prior    gaussian    0.37      0.485     0.6
#"

export NUISANCE_B_TO_VPERPLL_FORM_FACTORS="
    --global-option form-factors KMPW2010

    --nuisance    B->K^*::F^V(0)@KMPW2010                                   ${N_SIGMAS}     --prior    log-gamma   0.24      0.36      0.59
    --nuisance    B->K^*::b^V_1@KMPW2010                        -6.0   5.4  ${N_SIGMAS}     --prior    log-gamma  -5.20     -4.80     -4.00
"

export NUISANCE_B_TO_VPARALL_FORM_FACTORS="
    --global-option form-factors KMPW2010

    --nuisance    B->K^*::F^A1(0)@KMPW2010                                  ${N_SIGMAS}     --prior    log-gamma   0.15      0.25      0.41
    --nuisance    B->K^*::b^A1_1@KMPW2010                       -2.06  5.4  ${N_SIGMAS}     --prior    log-gamma  -0.46     +0.34     +1.20
    --nuisance    B->K^*::F^A2(0)@KMPW2010                                  ${N_SIGMAS}     --prior    log-gamma   0.13      0.23      0.42
    --nuisance    B->K^*::b^A2_1@KMPW2010                       -4.9   5.4  ${N_SIGMAS}     --prior    log-gamma  -2.20     -0.85     +2.03
"

export NUISANCE_B_TO_VPERPLL_SUBLEADING="
    --nuisance    B->K^*ll::A_perp^L_uncertainty@LargeRecoil     0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_perp^R_uncertainty@LargeRecoil     0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
"

export NUISANCE_B_TO_VPERPLL_SUBLEADING_FLAT="
    --nuisance    B->K^*ll::A_perp^L_uncertainty@LargeRecoil     0.55  1.45 --prior flat
    --nuisance    B->K^*ll::A_perp^R_uncertainty@LargeRecoil     0.55  1.45 --prior flat
"

export NUISANCE_B_TO_VPERPLL_SUBLEADING_WIDE="
    --nuisance    B->K^*ll::A_perp^L_uncertainty@LargeRecoil     0.2   1.8  ${N_SIGMAS}     --prior    gaussian    0.55      1.0       1.45
    --nuisance    B->K^*ll::A_perp^R_uncertainty@LargeRecoil     0.2   1.8  ${N_SIGMAS}     --prior    gaussian    0.55      1.0       1.45
"

export NUISANCE_B_TO_VPARALL_SUBLEADING="
    --nuisance    B->Vll::Lambda_0@LowRecoil                    -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->Vll::Lambda_pa@LowRecoil                   -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->Vll::Lambda_pp@LowRecoil                   -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->K^*ll::A_0^L_uncertainty@LargeRecoil        0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_0^R_uncertainty@LargeRecoil        0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_par^L_uncertainty@LargeRecoil      0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_par^R_uncertainty@LargeRecoil      0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
"
export NUISANCE_B_TO_VPARALL_SUBLEADING_WIDE="
    --nuisance    B->Vll::Lambda_0@LowRecoil                    -0.8   0.8  ${N_SIGMAS}     --prior    gaussian   -0.45      0.0       0.45
    --nuisance    B->Vll::Lambda_pa@LowRecoil                   -0.8   0.8  ${N_SIGMAS}     --prior    gaussian   -0.45      0.0       0.45
    --nuisance    B->Vll::Lambda_pp@LowRecoil                   -0.8   0.8  ${N_SIGMAS}     --prior    gaussian   -0.45      0.0       0.45
    --nuisance    B->K^*ll::A_0^L_uncertainty@LargeRecoil        0.2   1.8  ${N_SIGMAS}     --prior    gaussian    0.55      1.0       1.45
    --nuisance    B->K^*ll::A_0^R_uncertainty@LargeRecoil        0.2   1.8  ${N_SIGMAS}     --prior    gaussian    0.55      1.0       1.45
    --nuisance    B->K^*ll::A_par^L_uncertainty@LargeRecoil      0.2   1.8  ${N_SIGMAS}     --prior    gaussian    0.55      1.0       1.45
    --nuisance    B->K^*ll::A_par^R_uncertainty@LargeRecoil      0.2   1.8  ${N_SIGMAS}     --prior    gaussian    0.55      1.0       1.45
"

export NUISANCE_B_TO_VPARALL_SUBLEADING_FLAT="
    --nuisance    B->Vll::Lambda_0@LowRecoil                    -0.45  0.45 --prior    flat
    --nuisance    B->Vll::Lambda_pa@LowRecoil                   -0.45  0.45 --prior    flat
    --nuisance    B->Vll::Lambda_pp@LowRecoil                   -0.45  0.45 --prior    flat
    --nuisance    B->K^*ll::A_0^L_uncertainty@LargeRecoil        0.55  1.45 --prior    flat
    --nuisance    B->K^*ll::A_0^R_uncertainty@LargeRecoil        0.55  1.45 --prior    flat
    --nuisance    B->K^*ll::A_par^L_uncertainty@LargeRecoil      0.55  1.45 --prior    flat
    --nuisance    B->K^*ll::A_par^R_uncertainty@LargeRecoil      0.55  1.45 --prior    flat
"

# uncertainty of F^p averaged from KMPW2010 and BZ2004
export NUISANCE_B_TO_PLL_FORM_FACTORS="
    --global-option form-factors KMPW2010

    --nuisance    B->K::F^p(0)@KMPW2010                          0.10  0.49 ${N_SIGMAS}     --prior    gaussian    0.29      0.34      0.39
    --nuisance    B->K::b^p_1@KMPW2010                          -6.9   0.6  ${N_SIGMAS}     --prior    log-gamma  -3.7      -2.1      -1.2
"

# results as quoted in KMPW2010
export NUISANCE_B_TO_PLL_FORM_FACTORS_PLUS="
    --global-option form-factors KMPW2010

    --nuisance    B->K::F^p(0)@KMPW2010                          0.10  0.49 ${N_SIGMAS}     --prior    gaussian    0.32      0.34      0.39
    --nuisance    B->K::b^p_1@KMPW2010                          -6.9   0.6  ${N_SIGMAS}     --prior    log-gamma  -3.7      -2.1      -1.2
"

export NUISANCE_B_TO_PLL_FORM_FACTORS_SCALAR="
    --global-option form-factors KMPW2010

    --nuisance    B->K::b^0_1@KMPW2010                          -7.0   1.9  ${N_SIGMAS}     --prior    log-gamma  -5.2      -4.3      -3.5
"

export NUISANCE_B_TO_PLL_FORM_FACTORS_TENSOR="
    --global-option form-factors KMPW2010

    --nuisance    B->K::F^t(0)@KMPW2010                          0.30  0.54 ${N_SIGMAS}     --prior    gaussian    0.36      0.39      0.44
    --nuisance    B->K::b^t_1@KMPW2010                          -8.2   0.8  ${N_SIGMAS}     --prior    log-gamma  -4.2      -2.2      -1.2
"

export NUISANCE_B_TO_PLL_FORM_FACTORS_ALL="
    ${NUISANCE_B_TO_PLL_FORM_FACTORS_PLUS}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS_SCALAR}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS_TENSOR}
"

export NUISANCE_B_TO_PLL_SUBLEADING="
    --nuisance    B->Pll::Lambda_pseudo@LowRecoil               -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->Pll::Lambda_pseudo@LargeRecoil             -1.0   1.0  ${N_SIGMAS}     --prior    gaussian   -0.50      0.0       0.50
"

export NUISANCE_B_TO_PLL_SUBLEADING_WIDE="
    --nuisance    B->Pll::Lambda_pseudo@LowRecoil               -0.8   0.8  ${N_SIGMAS}     --prior    gaussian   -0.45      0.0       0.45
    --nuisance    B->Pll::Lambda_pseudo@LargeRecoil             -2.0   2.0  ${N_SIGMAS}     --prior    gaussian   -1.50      0.0       1.50
"

export NUISANCE_B_TO_PLL_SUBLEADING_FLAT="
    --nuisance    B->Pll::Lambda_pseudo@LowRecoil               -0.45  0.45 --prior    flat
    --nuisance    B->Pll::Lambda_pseudo@LargeRecoil             -1.00  1.00 --prior    flat
"

export NUISANCE_B_TO_XS_HQE="
    --nuisance    B->B::mu_pi^2@1GeV                             0.0   2.0  ${N_SIGMAS}     --prior    gaussian    0.35      0.45      0.55
    --nuisance    B->B::mu_G^2@1GeV                              0.0   2.0  ${N_SIGMAS}     --prior    log-gamma   0.33      0.35      0.38
"

export NUISANCE_all="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_DECAY_CONSTANTS}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS}
    ${NUISANCE_B_TO_PLL_SUBLEADING}
    ${NUISANCE_B_TO_XS_HQE}
"

export NUISANCE_allnolhcbfl=${NUISANCE_all}

export NUISANCE_allnobsmumu="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS}
    ${NUISANCE_B_TO_PLL_SUBLEADING}
    ${NUISANCE_B_TO_XS_HQE}
"

export NUISANCE_allnovll="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_DECAY_CONSTANTS}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS}
    ${NUISANCE_B_TO_PLL_SUBLEADING}
    ${NUISANCE_B_TO_XS_HQE}
"

export NUISANCE_allnoxsll=$NUISANCE_all

export NUISANCE_incl="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_B_TO_XS_HQE}
"

export NUISANCE_inclnoxsll="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_B_TO_XS_HQE}
"

export NUISANCE_excl="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_DECAY_CONSTANTS}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS}
    ${NUISANCE_B_TO_PLL_SUBLEADING}
"

export NUISANCE_bsgamma="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING}
"

export NUISANCE_bsmumu="
    ${NUISANCE_CKM}
    ${NUISANCE_DECAY_CONSTANTS}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS}
"

export NUISANCE_posthep13=${NUISANCE_all}
export NUISANCE_posthep13hpqcd=${NUISANCE_all}

export NUISANCE_posthep13noFLBabarAtlas=${NUISANCE_all}
export NUISANCE_posthep13hpqcdnoFLBabarAtlas=${NUISANCE_all}

export NUISANCE_posthep13noBKstargamma=${NUISANCE_all}
export NUISANCE_posthep13hpqcdnoBKstargamma=${NUISANCE_all}

export NUISANCE_posthep13hpqcdSLflat="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_DECAY_CONSTANTS}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING_FLAT}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING_FLAT}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS}
    ${NUISANCE_B_TO_PLL_SUBLEADING_FLAT}
    ${NUISANCE_B_TO_XS_HQE}
"

export NUISANCE_posthep13wide="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_DECAY_CONSTANTS}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING_WIDE}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING_WIDE}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS}
    ${NUISANCE_B_TO_PLL_SUBLEADING_WIDE}
    ${NUISANCE_B_TO_XS_HQE}
"

export NUISANCE_QUIMBASE="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_DECAY_CONSTANTS}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING}
    ${NUISANCE_B_TO_XS_HQE}
"
export NUISANCE_QUIMBASE_WIDE="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_DECAY_CONSTANTS}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING_WIDE}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING_WIDE}
    ${NUISANCE_B_TO_XS_HQE}
"

export NUISANCE_quim1=${NUISANCE_QUIMBASE}
export NUISANCE_quim2=${NUISANCE_QUIMBASE}
export NUISANCE_quim3=${NUISANCE_QUIMBASE}
export NUISANCE_quim1wide=${NUISANCE_QUIMBASE_WIDE}

export NUISANCE_uncVLL="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING}
"
export NUISANCE_uncVLLhpqcd=${NUISANCE_uncVLL}

export NUISANCE_uncPLL="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS}
    ${NUISANCE_B_TO_PLL_SUBLEADING}
"

export NUISANCE_uncVLLflat="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING_FLAT}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING_FLAT}
"

export NUISANCE_uncVLLwide="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_SUBLEADING_WIDE}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_SUBLEADING_WIDE}
"

export NUISANCE_uncINCL="
    ${NUISANCE_CKM}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_B_TO_XS_HQE}
"

export NUISANCE_uncFF="
    ${NUISANCE_B_TO_PLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPERPLL_FORM_FACTORS}
    ${NUISANCE_B_TO_VPARALL_FORM_FACTORS}
"
export NUISANCE_uncFFhpqcd=${NUISANCE_uncFF}

export NUISANCE_ckm14tensor="
    ${NUISANCE_CKM_posthep13}
    ${NUISANCE_QUARK_MASSES}
    ${NUISANCE_DECAY_CONSTANTS_flag13}
    ${NUISANCE_B_TO_PLL_FORM_FACTORS_ALL}
    ${NUISANCE_B_TO_PLL_SUBLEADING}
"
