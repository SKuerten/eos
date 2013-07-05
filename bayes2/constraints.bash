# vim: set sts=4 et :

export CONSTRAINTS_B_TO_XSGAMMA="
    --global-option model WilsonScan
    --global-option scan-mode cartesian

    --constraint B->X_sgamma::BR[1.8]@BaBar-2012
    --constraint B->X_sgamma::E_1[1.8]+E_2[1.8]@BaBar-2012
    --constraint B->X_sgamma::E_1[1.8]+E_2[1.8]@Belle-2008
"

export CONSTRAINTS_B_TO_KSTARGAMMA="
    --global-option model WilsonScan
    --global-option scan-mode cartesian

    --global-option form-factors KMPW2010

    --constraint B^0->K^*0gamma::BR@CLEO-2000
    --constraint B^0->K^*0gamma::BR@Belle-2004
    --constraint B^0->K^*0gamma::BR@BaBar-2009
    --constraint B^0->K^*0gamma::S_K+C_K@Belle-2006
    --constraint B^0->K^*0gamma::S_K+C_K@BaBar-2008
"

export CONSTRAINTS_BS_TO_MUMU="
    --global-option model WilsonScan
    --global-option scan-mode cartesian

    --constraint B^0_s->mu^+mu^-::BR_limit@LHCb-Nov-2012
"

export CONSTRAINTS_B_TO_XSLL="
    --global-option model WilsonScan
    --global-option scan-mode cartesian

    --constraint B->X_sll::BR[1.0,6.0]@BaBar-2004A
    --constraint B->X_sll::BR[1.0,6.0]@Belle-2005A
"

export CONSTRAINTS_B_TO_PLL_LARGE_RECOIL="
    --global-option model WilsonScan
    --global-option scan-mode cartesian

    --global-option form-factors KMPW2010

    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@Belle-2009
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@BaBar-2012
    --constraint B^+->K^+mu^+mu^-::BR[1.00,6.00]@LHCb-2012
"
export CONSTRAINTS_B_TO_PLL_LOW_RECOIL="
    --global-option model WilsonScan
    --global-option scan-mode cartesian

    --global-option form-factors KMPW2010

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

export CONSTRAINTS_B_TO_VLL_LARGE_RECOIL="
    --global-option model WilsonScan
    --global-option scan-mode cartesian

    --global-option form-factors KMPW2010

    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::BR[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::F_L[1.00,6.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^2[1.00,6.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^2[1.00,6.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_T^re[1.00,6.00]@LHCb-2013
"
#--constraint B^0->K^*0mu^+mu^-::S_9[1.00,6.00]@LHCb-2013

export CONSTRAINTS_B_TO_VLL_LOW_RECOIL="
    --global-option model WilsonScan
    --global-option scan-mode cartesian

    --global-option form-factors KMPW2010

    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[14.21,16.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::BR[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[14.18,16.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_T^2[14.18,16.00]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^2[14.18,16.00]@LHCb-2013

    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.21]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.21]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::BR[16.00,19.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.21]@Belle-2009
    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.21]@BaBar-2012
    --constraint B^0->K^*0mu^+mu^-::F_L[16.00,19.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_T^2[16.00,19.21]@CDF-2012
    --constraint B^0->K^*0mu^+mu^-::A_T^2[16.00,19.00]@LHCb-2013
    --constraint B^0->K^*0mu^+mu^-::A_T^re[16.00,19.00]@LHCb-2013
"
##    --constraint B^0->K^*0mu^+mu^-::A_T^re[14.18,16.00]@LHCb-2013
##    --constraint B^0->K^*0mu^+mu^-::S_9[14.18,16.00]@LHCb-2013
##    --constraint B^0->K^*0mu^+mu^-::S_9[16.00,19.00]@LHCb-2013

export CONSTRAINTS_ALL="
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_BS_TO_MUMU}
    ${CONSTRAINTS_B_TO_XSLL}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LOW_RECOIL}
"

export CONSTRAINTS_EXCL="
    ${CONSTRAINTS_B_TO_KSTARGAMMA}
    ${CONSTRAINTS_BS_TO_MUMU}
    ${CONSTRAINTS_B_TO_PLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_PLL_LOW_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LARGE_RECOIL}
    ${CONSTRAINTS_B_TO_VLL_LOW_RECOIL}
"

export CONSTRAINTS_INCL="
    ${CONSTRAINTS_B_TO_XSGAMMA}
    ${CONSTRAINTS_B_TO_XSLL}
"
