# vim: set sts=4 et :

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

# From PDG 2012
export NUISANCE_QUARK_MASSES="
    --nuisance    mass::c                                        0.0   2.0  ${N_SIGMAS}     --prior    gaussian    1.25      1.275     1.30
    --nuisance    mass::b(MSbar)                                            ${N_SIGMAS}     --prior    gaussian    4.15      4.18      4.21
"

# From Lattice Averages, June 2013
export NUISANCE_DECAY_CONSTANTS="
    --nuisance    decay-constant::B_d                                       ${N_SIGMAS}     --prior    gaussian    0.1859    0.1906    0.1953
    --nuisance    decay-constant::B_s                                       ${N_SIGMAS}     --prior    gaussian    0.2226    0.2276    0.2326
    --nuisance    decay-constant::K_d                                       ${N_SIGMAS}     --prior    gaussian    0.1549    0.1561    0.1573
"

# From before
export NUISANCE_B_TO_VLL_HADRONICS="
    --nuisance    B->K^*::f_Kstar_perp@2GeV                                 ${N_SIGMAS}     --prior    gaussian    0.168     0.173     0.178
    --nuisance    B->K^*::f_Kstar_par                                       ${N_SIGMAS}     --prior    gaussian    0.212     0.217     0.222
    --nuisance    lambda_B_p                                     0.1   0.9  ${N_SIGMAS}     --prior    gaussian    0.37      0.485     0.6
"

export NUISANCE_B_TO_VLL_FORM_FACTORS="
    --nuisance    B->K^*::F^V(0)@KMPW2010                        0.0   1.0  ${N_SIGMAS}     --prior    log-gamma   0.24      0.36      0.59
    --nuisance    B->K^*::b^V_1@KMPW2010                                    ${N_SIGMAS}     --prior    log-gamma  -5.20     -4.80     -4.00
    --nuisance    B->K^*::F^A1(0)@KMPW2010                       0.0   1.0  ${N_SIGMAS}     --prior    log-gamma   0.15      0.25      0.41
    --nuisance    B->K^*::b^A1_1@KMPW2010                                   ${N_SIGMAS}     --prior    log-gamma  -0.46     +0.34     +1.20
    --nuisance    B->K^*::F^A2(0)@KMPW2010                       0.0   1.0  ${N_SIGMAS}     --prior    log-gamma   0.13      0.23      0.42
    --nuisance    B->K^*::b^A2_1@KMPW2010                                   ${N_SIGMAS}     --prior    log-gamma  -2.20     -0.85     +2.03
"

export NUISANCE_B_TO_VLL_SUBLEADING="
    --nuisance    B->Vll::Lambda_0@LowRecoil                    -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->Vll::Lambda_pa@LowRecoil                   -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->Vll::Lambda_pp@LowRecoil                   -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->K^*ll::A_0^L_uncertainty@LargeRecoil        0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_0^R_uncertainty@LargeRecoil        0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_par^L_uncertainty@LargeRecoil      0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_par^R_uncertainty@LargeRecoil      0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_perp^L_uncertainty@LargeRecoil     0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
    --nuisance    B->K^*ll::A_perp^R_uncertainty@LargeRecoil     0.5   1.5  ${N_SIGMAS}     --prior    gaussian    0.85      1.0       1.15
"

export NUISANCE_B_TO_PLL_FORM_FACTORS="
    --nuisance    B->K::F^p(0)@KMPW2010                          0.28  0.49 ${N_SIGMAS}     --prior    log-gamma   0.32      0.34      0.39
    --nuisance    B->K::b^p_1@KMPW2010                          -6.9   0.6  ${N_SIGMAS}     --prior    log-gamma  -3.7      -2.1      -1.2
"

export NUISANCE_B_TO_PLL_SUBLEADING="
    --nuisance    B->Pll::Lambda_pseudo@LowRecoil               -0.5   0.5  ${N_SIGMAS}     --prior    gaussian   -0.15      0.0       0.15
    --nuisance    B->Pll::Lambda_pseudo@LargeRecoil             -1.0   1.0  ${N_SIGMAS}     --prior    gaussian   -0.50      0.0       0.50
"
