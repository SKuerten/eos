#####################
## building blocks ##
#####################
export CONSTRAINTS_BS_TO_MUMU_CKM14="
    --constraint B^0_s->mu^+mu^-::BR@CMS-LHCb-2014
"
#####################
## B -> K mu^+ mu^-##
#####################
export CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_BR="
    --global-option form-factors KMPW2010

    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[1.10,6.00]@LHCb-2014
"
export CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_BR="
    --global-option form-factors KMPW2010

    --constraint B^+->K^+mu^+mu^-::BR[14.18,16.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[16.00,22.86]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[15.00,22.00]@LHCb-2014
"
export CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_A_FB="
    --global-option form-factors KMPW2010

    --constraint B^+->K^+mu^+mu^-::A_FB[1.00,6.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::A_FB[1.10,6.00]@LHCb-2014
"
export CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_A_FB="
    --global-option form-factors KMPW2010

    --constraint B^+->K^+mu^+mu^-::A_FB[14.18,16.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::A_FB[16.00,22.86]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::A_FB[15.00,22.00]@LHCb-2014
"
export CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_F_H="
    --global-option form-factors KMPW2010

    --constraint B^+->K^+mu^+mu^-::F_H[1.10,6.00]@LHCb-2014
"
export CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_F_H="
    --global-option form-factors KMPW2010

    --constraint B^+->K^+mu^+mu^-::F_H[15.00,22.00]@LHCb-2014
"
export CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_A_FB_F_H="
    --global-option form-factors KMPW2010

    ${CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_A_FB}
    ${CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_F_H}
"
export CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_A_FB_F_H="
    --global-option form-factors KMPW2010

    ${CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_A_FB}
    ${CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_F_H}
"
export CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_BR_A_FB_F_H="
    --global-option form-factors KMPW2010

    ${CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_BR}
    ${CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_A_FB_F_H}
"
export CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_BR_A_FB_F_H="
    --global-option form-factors KMPW2010

    ${CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_BR}
    ${CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_A_FB_F_H}
"
export CONSTRAINTS_B_TO_K_FF_ALL="
    --global-option form-factors KMPW2010

    --constraint B->K::f_0+f_++f_T@HPQCD-2013A
"
export CONSTRAINTS_B_TO_KMUMU_A_CP="
    --global-option form-factors KMPW2010

    --constraint B^+->K^+mu^+mu^-::A_CP[1.10,6.00]@LHCb-2014
    --constraint B^+->K^+mu^+mu^-::A_CP[15.00,22.00]@LHCb-2014
"
#####################
## B -> K^* mu^+ mu^-
#####################
export CONSTRAINTS_B_TO_KSTARMUMU_LARGE_RECOIL_BR="
    --global-option form-factors BSZ2015

    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CMS-2013A
"
export CONSTRAINTS_B_TO_KSTARMUMU_LOW_RECOIL_BR="
    --global-option form-factors BSZ2015

    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.00]@CMS-2013A
"
export CONSTRAINTS_B_TO_KSTARMUMU_A_CP="
    --global-option form-factors BSZ2015

    --constraint B^0->K^*0mu^+mu^-::A_CP[1.10,6.00]@LHCb-2014
    --constraint B^0->K^*0mu^+mu^-::A_CP[15.00,19.00]@LHCb-2014
"
export CONSTRAINTS_B_TO_KSTARMUMU_A_FB="
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.00]@CMS-2013A
"
export CONSTRAINTS_B_TO_KSTAR_FF_LATTICE="
    --global-option form-factors BSZ2015

    --constraint B->K^*::V+A_0+A_1+A_12@HLMW-2015
    --constraint B->K^*::T_1+T_2+T_23@HLMW-2015
"
export CONSTRAINTS_B_TO_KSTAR_FF_LCSR="
    --global-option form-factors BSZ2015

    --constraint B->K^*::A_0+A_1+A_2+V+T_1+T_2+T_3@BSZ2015
"
export CONSTRAINTS_B_TO_KSTAR_FF_KINEMATIC_ENDPOINT="
    --global-option form-factors BSZ2015

    --constraint B->K^*::A_12(s_max)/A_1(s_max)@HLMW-2015
"
export CONSTRAINTS_B_TO_KSTAR_FF_ALL="
    ${CONSTRAINTS_B_TO_KSTAR_FF_LATTICE}
    ${CONSTRAINTS_B_TO_KSTAR_FF_LCSR}
"
#####################
## inclusive ##
#####################
export CONSTRAINTS_B_TO_XSGAMMA="
    --constraint B->X_sgamma::BR[1.8]@BaBar-2012
    --constraint B->X_sgamma::BR[1.8]@Belle-2009B
"
export CONSTRAINTS_B_MASS_SPLITTING="
    --constraint B^0::M_B^*-M_B@PDG-2012
"
#####################
## named data sets ##
#####################
export CONSTRAINTS_FH="
$CONSTRAINTS_B_TO_K_FF_ALL
$CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_F_H
$CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_F_H
"
export CONSTRAINTS_Bsmumu="
$CONSTRAINTS_BS_TO_MUMU_CKM14
"
export CONSTRAINTS_FH_Bsmumu="
$CONSTRAINTS_FH
$CONSTRAINTS_Bsmumu
"
export CONSTRAINTS_K_Bsmumu="
$CONSTRAINTS_B_TO_K_FF_ALL
$CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_BR_A_FB_F_H
$CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_BR_A_FB_F_H
$CONSTRAINTS_B_TO_KMUMU_A_CP
$CONSTRAINTS_Bsmumu
"
export CONSTRAINTS_K_KstarBR_no_ACP="
$CONSTRAINTS_B_TO_K_FF_ALL
$CONSTRAINTS_B_TO_KMUMU_LARGE_RECOIL_BR_A_FB_F_H
$CONSTRAINTS_B_TO_KMUMU_LOW_RECOIL_BR_A_FB_F_H

$CONSTRAINTS_B_TO_KSTAR_FF_ALL
$CONSTRAINTS_B_TO_KSTARMUMU_LOW_RECOIL_BR
$CONSTRAINTS_B_TO_KSTARMUMU_LARGE_RECOIL_BR
"
export CONSTRAINTS_K_KstarBR="
$CONSTRAINTS_K_KstarBR_no_ACP
$CONSTRAINTS_B_TO_KSTARMUMU_A_FB
$CONSTRAINTS_B_TO_KMUMU_A_CP
$CONSTRAINTS_B_TO_KSTARMUMU_A_CP
"
export CONSTRAINTS_K_KstarBR_no_ACP_Bsmumu="
$CONSTRAINTS_K_KstarBR_no_ACP
$CONSTRAINTS_Bsmumu
"
export CONSTRAINTS_K_KstarBR_Bsmumu="
$CONSTRAINTS_K_KstarBR
$CONSTRAINTS_Bsmumu
"
export CONSTRAINTS_K_KstarBR_Bsmumu_noSL=${CONSTRAINTS_K_KstarBR_Bsmumu}
export CONSTRAINTS_unc_K=$CONSTRAINTS_B_TO_K_FF_ALL
export CONSTRAINTS_unc_Kstar="
$CONSTRAINTS_B_TO_KSTAR_FF_ALL
"
# no constraints, just priors
export CONSTRAINTS_unc_Kstar_uncorr=
