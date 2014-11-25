echo "[constraints loaded]"

#####################
## building blocks ##
#####################

export CONSTRAINTS_BS_TO_MUMU_CKM14="
    --constraint B^0_s->mu^+mu^-::BR@CMS-LHCb-2014
"

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
    --constraint B^+->K^+mu^+mu^-::A_FB[1.10,6.00]@LHCb-2014
"

export CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_A_FB="
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
## named data sets ##
#####################

export CONSTRAINTS_FH="
${CONSTRAINTS_B_TO_PLL_FF_ALL}
${CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_F_H}
${CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_F_H}
"
