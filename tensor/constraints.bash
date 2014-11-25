echo "[constraints loaded]"

#####################
## building blocks ##
#####################

export CONSTRAINTS_BS_TO_MUMU_CKM14="
    --constraint B^0_s->mu^+mu^-::BR@CMS-LHCb-2014
"

#####################
## B -> K mu^+ mu^-##
#####################
export CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_BR="
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[1.10,6.00]@LHCb-2014
"

export CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_BR="
    --constraint B^+->K^+mu^+mu^-::BR[14.18,16.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[16.00,22.86]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[15.00,22.00]@LHCb-2014
"

export CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_A_FB="
    --constraint B^+->K^+mu^+mu^-::A_FB[1.00,6.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::A_FB[1.10,6.00]@LHCb-2014
"

export CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_A_FB="
    --constraint B^+->K^+mu^+mu^-::A_FB[14.18,16.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::A_FB[16.00,22.86]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::A_FB[15.00,22.00]@LHCb-2014
"

export CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_F_H="
    --constraint B^+->K^+mu^+mu^-::F_H[1.10,6.00]@LHCb-2014
"

export CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_F_H="
    --constraint B^+->K^+mu^+mu^-::F_H[15.00,22.00]@LHCb-2014
"

export CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_A_FB_F_H="
    ${CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_A_FB}
    ${CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_F_H}
"

export CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_A_FB_F_H="
    ${CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_A_FB}
    ${CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_F_H}
"

export CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_BR_A_FB_F_H="
    ${CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_BR}
    ${CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_A_FB_F_H}
"

export CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_BR_A_FB_F_H="
    ${CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_BR}
    ${CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_A_FB_F_H}
"

export CONSTRAINTS_B_TO_PLL_FF_ALL="
    --constraint B->K::f_0+f_++f_T@HPQCD-2013A
"

#####################
## B -> K^* mu^+ mu^-
#####################
export CONSTRAINTS_B_TO_VMUMU_LARGE_RECOIL_BR="
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CMS-2013A
"

export CONSTRAINTS_B_TO_VMUMU_LOW_RECOIL_BR="
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.00]@CMS-2013A
"

export CONSTRAINTS_B_TO_VLL_FF="
    --kinematics s 0.0 --observable-prior B->K^*::V(s)/A_1(s) 0.93 1.33 1.73
    --kinematics s 0.0 --observable-prior B->K^*ll::xi_para(s)@LargeRecoil 0.08 0.10 0.13
"

export CONSTRAINTS_B_TO_VLL_FF_HPQCD="
    --constraint B->K^*::V@HPQCD-2013B
    --constraint B->K^*::A_1@HPQCD-2013B
    --constraint B->K^*::A_12@HPQCD-2013B
"

export CONSTRAINTS_B_TO_VLL_FF_RATIOS="
--constraint B->K^*::V(0)/A_1(0)@HHSZ2013
--constraint B->K^*::xi_para(0)@KMPW2010
"

#####################
## named data sets ##
#####################

export CONSTRAINTS_FH="
$CONSTRAINTS_B_TO_PLL_FF_ALL
$CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_F_H
$CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_F_H
"

export CONSTRAINTS_K_Kstar_BR="
$CONSTRAINTS_B_TO_PLL_FF_ALL
$CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_BR_A_FB_F_H
$CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_BR_A_FB_F_H

$CONSTRAINTS_B_TO_VMUMU_LOW_RECOIL_BR
$CONSTRAINTS_B_TO_VMUMU_LARGE_RECOIL_BR
$CONSTRAINTS_B_TO_VLL_FF_RATIOS
"
