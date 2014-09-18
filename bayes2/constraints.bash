# vim: set sts=4 et :

echo "[constraints loaded]"

##################
## Our Datasets ##
##################

export CONSTRAINTS_B_TO_XSGAMMA="
    --constraint B->X_sgamma::BR[1.8]@BaBar-2012
    --constraint B->X_sgamma::BR[1.8]@Belle-2009B
"
#    --constraint B->X_sgamma::E_1[1.8]+E_2[1.8]@BaBar-2012
#    --constraint B->X_sgamma::BR[1.8]+E_1[1.8]+E_2[1.8]@Belle-2009B

export CONSTRAINTS_B_TO_KSTARGAMMA="
    --constraint B^0->K^*0gamma::BR@CLEO-2000
    --constraint B^0->K^*0gamma::BR@Belle-2004
    --constraint B^0->K^*0gamma::BR@BaBar-2009
    --constraint B^0->K^*0gamma::S_K+C_K@Belle-2006
    --constraint B^0->K^*0gamma::S_K+C_K@BaBar-2008
"

export CONSTRAINTS_BS_TO_MUMU="
    --constraint B^0_s->mu^+mu^-::BR@CMS-2013B
    --constraint B^0_s->mu^+mu^-::BR@LHCb-2013D
"

export CONSTRAINTS_BS_TO_MUMU_POSTHEP13="
    --constraint B^0_s->mu^+mu^-::BR@CMS-2013B
    --constraint B^0_s->mu^+mu^-::BR@LHCb-2013D
"

export CONSTRAINTS_BS_TO_MUMU_CKM14="
    --constraint B^0_s->mu^+mu^-::BR@CMS-LHCb-2014
"

export CONSTRAINTS_B_TO_XSLL="
    --constraint B->X_sll::BR[1.0,6.0]@BaBar-2004A
    --constraint B->X_sll::BR[1.0,6.0]@Belle-2005A
"

export CONSTRAINTS_B_TO_PLL_LARGE_RECOIL="
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@Belle-2009
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@BaBar-2012
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@LHCb-2012
"
export CONSTRAINTS_B_TO_PLL_LOW_RECOIL="
    --constraint B^+->K^+mu^+mu^-::BR[14.18,16.00]@Belle-2009
    --constraint B^+->K^+mu^+mu^-::BR[14.18,16.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[14.21,16.00]@BaBar-2012
    --constraint B^+->K^+mu^+mu^-::BR[14.18,16.00]@LHCb-2012

    --constraint B^+->K^+mu^+mu^-::BR[16.00,18.00]@LHCb-2012
    --constraint B^+->K^+mu^+mu^-::BR[18.00,22.00]@LHCb-2012

    --constraint B^+->K^+mu^+mu^-::BR[16.00,22.86]@Belle-2009
    --constraint B^+->K^+mu^+mu^-::BR[16.00,22.86]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[16.00,22.86]@BaBar-2012
"

export CONSTRAINTS_B_TO_PLL_LARGE_RECOIL_CKM14="
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@Belle-2009
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@BaBar-2012
    --constraint B^+->K^+mu^+mu^-::BR[1.10,6.00]@LHCb-2014
"

export CONSTRAINTS_B_TO_PLL_LOW_RECOIL_CKM14="
    --constraint B^+->K^+mu^+mu^-::BR[14.18,16.00]@Belle-2009
    --constraint B^+->K^+mu^+mu^-::BR[14.18,16.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[14.21,16.00]@BaBar-2012

    --constraint B^+->K^+mu^+mu^-::BR[15.00,22.00]@LHCb-2014

    --constraint B^+->K^+mu^+mu^-::BR[16.00,22.86]@Belle-2009
    --constraint B^+->K^+mu^+mu^-::BR[16.00,22.86]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[16.00,22.86]@BaBar-2012
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

export CONSTRAINTS_B_TO_PLL_FF="
    --constraint B->K::f_+@HPQCD-2013A
"

export CONSTRAINTS_B_TO_PLL_FF_ALL="
    --constraint B->K::f_0+f_++f_T@HPQCD-2013A
"

export CONSTRAINTS_B_TO_VLL_LARGE_RECOIL="
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^re[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@CMS-2013A
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@ATLAS-2013A

    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@CMS-2013A
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@ATLAS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_T^2[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^2[1.00,6.00]@LHCb-2013
"

export CONSTRAINTS_B_TO_VLL_LARGE_RECOIL_NO_FL_BABAR_ATLAS="
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^re[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@CMS-2013A
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@ATLAS-2013A

    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_T^2[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^2[1.00,6.00]@LHCb-2013
"

export CONSTRAINTS_B_TO_VLL_LARGE_RECOIL_NOLHCBFL="
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_T^re[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@CMS-2013A
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@ATLAS-2013A

    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@CMS-2013A
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@ATLAS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_T^2[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^2[1.00,6.00]@LHCb-2013
"

export CONSTRAINTS_B_TO_VLL_LOW_RECOIL="
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[14.21,16.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@CMS-2013A
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@ATLAS-2013A

    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@CMS-2013A
    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@ATLAS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_T^2[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^2[14.18,16.00]@LHCb-2013


    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.21]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.21]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.00]@CMS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.00]@CMS-2013A
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.00]@ATLAS-2013A

    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.21]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.21]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.00]@CMS-2013A
    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.00]@ATLAS-2013A

    --constraint B^0->K^*0mu^+mu^-::A_T^2[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^2[16.00,19.00]@LHCb-2013
"

export CONSTRAINTS_B_TO_VLL_POSTHEP13="
    --constraint B^0->K^*0mu^+mu^-::P'_4[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::P'_4[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::P'_4[16.00,19.00]@LHCb-2013

    --constraint B^0->K^*0mu^+mu^-::P'_5[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::P'_5[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::P'_5[16.00,19.00]@LHCb-2013

    --constraint B^0->K^*0mu^+mu^-::P'_6[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::P'_6[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::P'_6[16.00,19.00]@LHCb-2013
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

export CONSTRAINTS_B_MASS_SPLITTING="
    --constraint B^0::M_B^*-M_B@PDG-2012
"

#################################
## combinations of constraints ##
#################################

export CONSTRAINTS_all="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_BS_TO_MUMU}
    ${CONSTRAINTS_B_TO_XSLL}
    ${CONSTRAINTS_B_TO_PLL_FF}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_FF}
"

export CONSTRAINTS_posthep13="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_BS_TO_MUMU_POSTHEP13}
    ${CONSTRAINTS_B_TO_XSLL}
    ${CONSTRAINTS_B_TO_PLL_FF}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_POSTHEP13}
    ${CONSTRAINTS_B_TO_VLL_FF}
"
export CONSTRAINTS_posthep13wide=${CONSTRAINTS_posthep13}

export CONSTRAINTS_posthep13hpqcd="
    ${CONSTRAINTS_posthep13}
    ${CONSTRAINTS_B_TO_VLL_FF_HPQCD}
"

export CONSTRAINTS_posthep13hpqcdSLflat=${CONSTRAINTS_posthep13hpqcd}

export CONSTRAINTS_posthep13noFLBabarAtlas="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_BS_TO_MUMU_POSTHEP13}
    ${CONSTRAINTS_B_TO_XSLL}
    ${CONSTRAINTS_B_TO_PLL_FF}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LARGE_RECOIL_NO_FL_BABAR_ATLAS}
    ${CONSTRAINTS_B_TO_VLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_POSTHEP13}
    ${CONSTRAINTS_B_TO_VLL_FF}
    ${CONSTRAINTS_B_TO_VLL_FF_HPQCD}
"

export CONSTRAINTS_posthep13hpqcdnoFLBabarAtlas="
    ${CONSTRAINTS_posthep13noFLBabarAtlas}
    ${CONSTRAINTS_B_TO_VLL_FF_HPQCD}
"

export CONSTRAINTS_allnolhcbfl="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_BS_TO_MUMU}
    ${CONSTRAINTS_B_TO_XSLL}
    ${CONSTRAINTS_B_TO_PLL_FF}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LARGE_RECOIL_NOLHCBFL}
    ${CONSTRAINTS_B_TO_VLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_FF}
"

export CONSTRAINTS_allnobsmumu="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_B_TO_XSLL}
    ${CONSTRAINTS_B_TO_PLL_FF}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_FF}
"

export CONSTRAINTS_allnovll="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_B_TO_XSLL}
    ${CONSTRAINTS_BS_TO_MUMU}
    ${CONSTRAINTS_B_TO_PLL_FF}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
"

export CONSTRAINTS_allnoxsll="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_BS_TO_MUMU}
    ${CONSTRAINTS_B_TO_PLL_FF}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_FF}
"

export CONSTRAINTS_bsgamma="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
"

export CONSTRAINTS_bsmumu="
    ${CONSTRAINTS_BS_TO_MUMU}
"

export CONSTRAINTS_excl="
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_BS_TO_MUMU}
    ${CONSTRAINTS_B_TO_PLL_FF}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_FF}
"

export CONSTRAINTS_incl="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_XSLL}
"

export CONSTRAINTS_inclnoxsll="
    ${CONSTRAINTS_B_MASS_SPLITTING}
    ${CONSTRAINTS_B_TO_XSGAMMA}
"

#####################
## Quim's Datasets ##
#####################

# We only exclude the isospin asymmetry in B->K^*ll.
export CONSTRAINTS_QUIMBASE="
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_XSLL}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_BS_TO_MUMU_POSTHEP13}
    ${CONSTRAINTS_B_TO_VLL_POSTHEP13}
    ${CONSTRAINTS_B_TO_VLL_FF}
"

# quim1 = B->K^*ll, large recoil only
export CONSTRAINTS_quim1="
    ${CONSTRAINTS_QUIMBASE}
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_T^2[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_T^re[1.00,6.00]@LHCb-2013
"
export CONSTRAINTS_quim1wide=${CONSTRAINTS_quim1}

export CONSTRAINTS_uncVLL="${CONSTRAINTS_B_TO_VLL_FF}"
export CONSTRAINTS_uncVLLflat="${CONSTRAINTS_B_TO_VLL_FF}"
export CONSTRAINTS_uncVLLwide="${CONSTRAINTS_B_TO_VLL_FF}"

export CONSTRAINTS_uncVLLhpqcd="
    ${CONSTRAINTS_B_TO_VLL_FF}
    ${CONSTRAINTS_B_TO_VLL_FF_HPQCD}
"

export CONSTRAINTS_uncPLL="${CONSTRAINTS_B_TO_PLL_FF}"

export CONSTRAINTS_uncINCL="${CONSTRAINTS_B_MASS_SPLITTING}"

export CONSTRAINTS_uncFF="
    ${CONSTRAINTS_B_TO_PLL_FF}
    ${CONSTRAINTS_B_TO_VLL_FF}
"

export CONSTRAINTS_uncFFhpqcd="
    ${CONSTRAINTS_uncFF}
    ${CONSTRAINTS_B_TO_VLL_FF_HPQCD}
"

export CONSTRAINTS_posthep13wide=${CONSTRAINTS_posthep13}

export CONSTRAINTS_posthep13hpqcd="
    ${CONSTRAINTS_posthep13}
    ${CONSTRAINTS_B_TO_VLL_FF_HPQCD}
"

export CONSTRAINTS_ckm14tensor="
    ${CONSTRAINTS_BS_TO_MUMU_CKM14}
    ${CONSTRAINTS_B_TO_PLL_FF_ALL}
    ${CONSTRAINTS_B_TO_PMUMU_LARGE_RECOIL_BR_A_FB_F_H}
    ${CONSTRAINTS_B_TO_PMUMU_LOW_RECOIL_BR_A_FB_F_H}
"
