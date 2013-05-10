#!/bin/bash

PREFIX=lhcb-2012-02/
mkdir -p $PREFIX

## Generate differential data ##
## B^0 -> K^0 ll @ Large Recoil ##
echo "B^0 -> K^0 ll @ Large Recoil"
LAREOPT=",model=SM,form-factors=KMPW2010,l=mu,q=d"
./src/clients/eos-evaluate \
    --range s  0.05 8.95 178 --observable "B->Kll::dBR/ds@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->Kll::F_H(s)@LargeRecoil$LAREOPT" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --vary "B->K::F^p(0)@KMPW2010" \
    --vary "B->K::b^p_1@KMPW2010" \
    --vary "B->Pll::Lambda_pseudo@LargeRecoil" \
    --vary "mass::b(MSbar)" \
    --vary "mass::c" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}b0tok0ll-sm-diff-loq2.data

## B^- -> K^- ll @ Large Recoil ##
echo "B^- -> K^- ll @ Large Recoil"
LAREOPT=",model=SM,form-factors=KMPW2010,l=mu,q=u"
./src/clients/eos-evaluate \
    --range s  0.05 8.95 178 --observable "B->Kll::dBR/ds@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->Kll::F_H(s)@LargeRecoil$LAREOPT" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --vary "B->K::F^p(0)@KMPW2010" \
    --vary "B->K::b^p_1@KMPW2010" \
    --vary "B->Pll::Lambda_pseudo@LargeRecoil" \
    --vary "mass::b(MSbar)" \
    --vary "mass::c" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}bmtokmll-sm-diff-loq2.data

## B^0 -> K^0 ll @ Low Recoil ##
echo "B^0 -> K^0 ll @ Low Recoil"
LOREOPT=",model=SM,form-factors=KMPW2010,l=mu,q=d"
./src/clients/eos-evaluate \
    --range s 10.00 22.86 184 --observable "B->Kll::dBR/ds@LowRecoil$LOREOPT" \
    --range s 10.00 22.86 184 --observable "B->Kll::F_H(s)@LowRecoil$LOREOPT" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --vary "B->K::F^p(0)@KMPW2010" \
    --vary "B->K::b^p_1@KMPW2010" \
    --vary "B->Pll::Lambda_pseudo@LowRecoil" \
    --vary "mass::b(MSbar)" \
    --vary "mass::c" \
    --vary "mass::t(pole)" \
    --vary "B->Vll::Lambda_0@LowRecoil" \
    --vary "B->Vll::Lambda_pa@LowRecoil" \
    --vary "B->Vll::Lambda_pp@LowRecoil" \
    --vary "mu" \
    > ${PREFIX}b0tok0ll-sm-diff-hiq2.data

## B^- -> K^- ll @ Low Recoil ##
echo "B^- -> K^- ll @ Low Recoil"
LOREOPT=",model=SM,form-factors=KMPW2010,l=mu,q=u"
./src/clients/eos-evaluate \
    --range s 10.00 22.86 184 --observable "B->Kll::dBR/ds@LowRecoil$LOREOPT" \
    --range s 10.00 22.86 184 --observable "B->Kll::F_H(s)@LowRecoil$LOREOPT" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --vary "B->K::F^p(0)@KMPW2010" \
    --vary "B->K::b^p_1@KMPW2010" \
    --vary "B->Pll::Lambda_pseudo@LowRecoil" \
    --vary "mass::b(MSbar)" \
    --vary "mass::c" \
    --vary "mass::t(pole)" \
    --vary "B->Vll::Lambda_0@LowRecoil" \
    --vary "B->Vll::Lambda_pa@LowRecoil" \
    --vary "B->Vll::Lambda_pp@LowRecoil" \
    --vary "mu" \
    > ${PREFIX}bmtokmll-sm-diff-hiq2.data

## B^0 -> K^*0 ll @ Large Recoil ##
echo "B^0 -> K^*0 ll @ Large Recoil"
LAREOPT=",model=SM,form-factors=BZ2004,l=mu,q=d"
./src/clients/eos-evaluate \
    --range s  0.05 8.95 178 --observable "B->K^*ll::dBR/ds@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->K^*ll::F_L(s)@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->K^*ll::F_T(s)@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->K^*ll::A_FB(s)@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->K^*ll::A_T^2(s)@LargeRecoil$LAREOPT" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --vary "formfactors::xi_perp_uncertainty" \
    --vary "formfactors::xi_par_uncertainty" \
    --vary "mass::b(MSbar)" \
    --vary "mass::c" \
    --vary "mass::t(pole)" \
    --vary "B->K^*ll::A_0^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_0^R_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_par^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_par^R_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_perp^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_perp^R_uncertainty@LargeRecoil" \
    --vary "mu" \
    > ${PREFIX}b0tokstar0ll-sm-diff-loq2.data

## B^- -> K^*- ll @ Large Recoil ##
echo "B^- -> K^*- ll @ Large Recoil"
LAREOPT=",model=SM,form-factors=BZ2004,l=mu,q=u"
./src/clients/eos-evaluate \
    --range s  0.05 8.95 178 --observable "B->K^*ll::dBR/ds@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->K^*ll::F_L(s)@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->K^*ll::F_T(s)@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->K^*ll::A_FB(s)@LargeRecoil$LAREOPT" \
    --range s  0.05 8.95 178 --observable "B->K^*ll::A_T^2(s)@LargeRecoil$LAREOPT" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --vary "formfactors::xi_perp_uncertainty" \
    --vary "formfactors::xi_par_uncertainty" \
    --vary "mass::b(MSbar)" \
    --vary "mass::c" \
    --vary "mass::t(pole)" \
    --vary "B->K^*ll::A_0^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_0^R_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_par^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_par^R_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_perp^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_perp^R_uncertainty@LargeRecoil" \
    --vary "mu" \
    > ${PREFIX}bmtokstarmll-sm-diff-loq2.data

## B^0 -> K^*0 ll @ Low Recoil ##
echo "B^0 -> K^*0 ll @ Low Recoil"
LOREOPT=",model=SM,form-factors=BZ2004,l=mu,q=d"
./src/clients/eos-evaluate \
    --range s 10.00 19.2 184 --observable "B->K^*ll::dBR/ds@LowRecoil$LOREOPT" \
    --range s 10.00 19.2 184 --observable "B->K^*ll::F_L(s)@LowRecoil$LOREOPT" \
    --range s 10.00 19.2 184 --observable "B->K^*ll::F_T(s)@LowRecoil$LOREOPT" \
    --range s 10.00 19.2 184 --observable "B->K^*ll::A_FB(s)@LowRecoil$LOREOPT" \
    --range s 10.00 19.2 184 --observable "B->K^*ll::A_T^2(s)@LowRecoil$LOREOPT" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --vary "B->K^*::v_uncertainty@BZ2004" \
    --vary "B->K^*::a1_uncertainty@BZ2004" \
    --vary "B->K^*::a2_uncertainty@BZ2004" \
    --vary "mass::b(MSbar)" \
    --vary "mass::c" \
    --vary "mass::t(pole)" \
    --vary "B->Vll::Lambda_0@LowRecoil" \
    --vary "B->Vll::Lambda_pa@LowRecoil" \
    --vary "B->Vll::Lambda_pp@LowRecoil" \
    --vary "mu" \
    > ${PREFIX}b0tokstar0ll-sm-diff-hiq2.data

## B^- -> K^*- ll @ Low Recoil ##
echo "B^0 -> K^*0 ll @ Low Recoil"
LOREOPT=",model=SM,form-factors=BZ2004,l=mu,q=u"
./src/clients/eos-evaluate \
    --range s 10.00 19.2 184 --observable "B->K^*ll::dBR/ds@LowRecoil$LOREOPT" \
    --range s 10.00 19.2 184 --observable "B->K^*ll::F_L(s)@LowRecoil$LOREOPT" \
    --range s 10.00 19.2 184 --observable "B->K^*ll::F_T(s)@LowRecoil$LOREOPT" \
    --range s 10.00 19.2 184 --observable "B->K^*ll::A_FB(s)@LowRecoil$LOREOPT" \
    --range s 10.00 19.2 184 --observable "B->K^*ll::A_T^2(s)@LowRecoil$LOREOPT" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --vary "B->K^*::v_uncertainty@BZ2004" \
    --vary "B->K^*::a1_uncertainty@BZ2004" \
    --vary "B->K^*::a2_uncertainty@BZ2004" \
    --vary "mass::b(MSbar)" \
    --vary "mass::c" \
    --vary "mass::t(pole)" \
    --vary "B->Vll::Lambda_0@LowRecoil" \
    --vary "B->Vll::Lambda_pa@LowRecoil" \
    --vary "B->Vll::Lambda_pp@LowRecoil" \
    --vary "mu" \
    > ${PREFIX}bmtokstarmll-sm-diff-hiq2.data

## Generate binned data ##
## B^0 -> K^0 ll @ Low Recoil ##
echo "B^0 -> K^0 ll @ Low Recoil"
BINOPT=",model=SM,form-factors=KMPW2010,l=mu,q=d"
./src/clients/eos-evaluate \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 22.86 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 22.86 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "B->K::F^p(0)@KMPW2010" \
    --vary "B->K::b^p_1@KMPW2010" \
    --budget "SL" \
    --vary "B->Pll::Lambda_pseudo@LowRecoil" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}b0tok0ll-sm-binned-hiq2.data

## B^- -> K^- ll @ Low Recoil ##
echo "B^- -> K^- ll @ Low Recoil"
BINOPT=",model=SM,form-factors=KMPW2010,l=mu,q=u"
./src/clients/eos-evaluate \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 22.86 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 22.86 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "B->K::F^p(0)@KMPW2010" \
    --vary "B->K::b^p_1@KMPW2010" \
    --budget "SL" \
    --vary "B->Pll::Lambda_pseudo@LowRecoil" \
    --budget "C" \
    --vary "mass::c" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}bmtokmll-sm-binned-hiq2.data

## B^0 -> K^0 ll @ Large Recoil ##
echo "B^0 -> K^0 ll @ Large Recoil"
BINOPT=",model=SM,form-factors=KMPW2010,l=mu,q=d"
./src/clients/eos-evaluate \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->Kll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->Kll::F_H@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->Kll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->Kll::F_H@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->Kll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->Kll::F_H@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->Kll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->Kll::F_H@LargeRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "B->K::F^p(0)@KMPW2010" \
    --vary "B->K::b^p_1@KMPW2010" \
    --budget "SL" \
    --vary "B->Pll::Lambda_pseudo@LargeRecoil" \
    --budget "C" \
    --vary "mass::c" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}b0tok0ll-sm-binned-loq2.data

## B^- -> K^- ll @ Large Recoil ##
echo "B^- -> K^- ll @ Large Recoil"
BINOPT=",model=SM,form-factors=KMPW2010,l=mu,q=u"
./src/clients/eos-evaluate \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->Kll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->Kll::F_H@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->Kll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->Kll::F_H@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->Kll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->Kll::F_H@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->Kll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->Kll::F_H@LargeRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "B->K::F^p(0)@KMPW2010" \
    --vary "B->K::b^p_1@KMPW2010" \
    --budget "SL" \
    --vary "B->Pll::Lambda_pseudo@LargeRecoil" \
    --budget "C" \
    --vary "mass::c" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}bmtokmll-sm-binned-loq2.data

## B^0 -> K^*0 ll @ Low Recoil ##
echo "B^0 -> K^*0 ll @ Low Recoil"
BINOPT=",model=SM,form-factors=BZ2004,l=mu,q=d"
./src/clients/eos-evaluate \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::A_FB@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::F_L@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::F_T@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::A_T^2@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::A_FB@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::F_L@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::F_T@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::A_T^2@LowRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "B->K^*::v_uncertainty@BZ2004" \
    --vary "B->K^*::a1_uncertainty@BZ2004" \
    --vary "B->K^*::a2_uncertainty@BZ2004" \
    --budget "SL" \
    --vary "B->Vll::Lambda_0@LowRecoil" \
    --vary "B->Vll::Lambda_pa@LowRecoil" \
    --vary "B->Vll::Lambda_pp@LowRecoil" \
    --budget "C" \
    --vary "mass::c" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}b0tokstar0ll-sm-binned-hiq2.data

## B^- -> K^*- ll @ Low Recoil ##
echo "B^- -> K^*- ll @ Low Recoil"
BINOPT=",model=SM,form-factors=BZ2004,l=mu,q=u"
./src/clients/eos-evaluate \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::A_FB@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::F_L@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::F_T@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->K^*ll::A_T^2@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::A_FB@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::F_L@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::F_T@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 19.21 --observable "B->K^*ll::A_T^2@LowRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "B->K^*::v_uncertainty@BZ2004" \
    --vary "B->K^*::a1_uncertainty@BZ2004" \
    --vary "B->K^*::a2_uncertainty@BZ2004" \
    --budget "SL" \
    --vary "B->Vll::Lambda_0@LowRecoil" \
    --vary "B->Vll::Lambda_pa@LowRecoil" \
    --vary "B->Vll::Lambda_pp@LowRecoil" \
    --budget "C" \
    --vary "mass::c" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}bmtokstarmll-sm-binned-hiq2.data

## B^0 -> K^*0 ll @ Large Recoil ##
echo "B^0 -> K^*0 ll @ Large Recoil"
BINOPT=",model=SM,form-factors=BZ2004,l=mu,q=d"
./src/clients/eos-evaluate \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::A_FB@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::F_L@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::F_T@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::A_T^2@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::A_FB@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::F_L@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::F_T@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::A_T^2@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::A_FB@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::F_L@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::F_T@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::A_T^2@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::A_FB@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::F_L@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::F_T@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::A_T^2@LargeRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "formfactors::xi_perp_uncertainty" \
    --vary "formfactors::xi_par_uncertainty" \
    --budget "SL" \
    --vary "B->K^*ll::A_0^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_0^R_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_par^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_par^R_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_perp^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_perp^R_uncertainty@LargeRecoil" \
    --budget "C" \
    --vary "mass::c" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}b0tokstar0ll-sm-binned-loq2.data

echo "B^- -> K^*- ll @ Large Recoil"
BINOPT=",model=SM,form-factors=BZ2004,l=mu,q=u"
./src/clients/eos-evaluate \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::A_FB@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::F_L@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::F_T@LargeRecoil$BINOPT" \
    --kinematics s_min  1.000 --kinematics s_max  6.00 --observable "B->K^*ll::A_T^2@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::A_FB@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::F_L@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::F_T@LargeRecoil$BINOPT" \
    --kinematics s_min  0.045 --kinematics s_max  2.00 --observable "B->K^*ll::A_T^2@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::A_FB@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::F_L@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::F_T@LargeRecoil$BINOPT" \
    --kinematics s_min  2.000 --kinematics s_max  4.30 --observable "B->K^*ll::A_T^2@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::BRavg@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::A_FB@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::F_L@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::F_T@LargeRecoil$BINOPT" \
    --kinematics s_min  4.300 --kinematics s_max  8.68 --observable "B->K^*ll::A_T^2@LargeRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "formfactors::xi_perp_uncertainty" \
    --vary "formfactors::xi_par_uncertainty" \
    --budget "SL" \
    --vary "B->K^*ll::A_0^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_0^R_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_par^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_par^R_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_perp^L_uncertainty@LargeRecoil" \
    --vary "B->K^*ll::A_perp^R_uncertainty@LargeRecoil" \
    --budget "C" \
    --vary "mass::c" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}bmtokstarmll-sm-binned-loq2.data

# Generate tarball
echo "Generating tarball"
tar zcf ${PREFIX%/}.tar.gz ${PREFIX}
