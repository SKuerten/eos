#!/bin/bash

PREFIX=lhcb-2012-09/
mkdir -p $PREFIX

## Generate binned data ##
## B^0 -> K^0 ll @ Low Recoil ##
echo "B^0 -> K^0 ll @ Low Recoil"
BINOPT=",model=SM,form-factors=BZ2004v2,l=mu,q=d"
./src/clients/eos-evaluate \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 18.00 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 18.00 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --kinematics s_min  18.00 --kinematics s_max 22.86 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  18.00 --kinematics s_max 22.86 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 22.86 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 22.86 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "B->K::fp_uncertainty@BZ2004v2" \
    --budget "SL" \
    --vary "B->Pll::Lambda_pseudo@LowRecoil" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}b0tok0ll-sm-binned-hiq2.data

## B^- -> K^- ll @ Low Recoil ##
echo "B^- -> K^- ll @ Low Recoil"
BINOPT=",model=SM,form-factors=BZ2004v2,l=mu,q=u"
./src/clients/eos-evaluate \
    --parameter "mass::K0" 0.49368 \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  14.18 --kinematics s_max 16.00 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 18.00 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 18.00 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --kinematics s_min  18.00 --kinematics s_max 22.90 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  18.00 --kinematics s_max 22.90 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 22.90 --observable "B->Kll::BRavg@LowRecoil$BINOPT" \
    --kinematics s_min  16.00 --kinematics s_max 22.90 --observable "B->Kll::F_H@LowRecoil$BINOPT" \
    --budget "CKM" \
    --vary "CKM::A" \
    --vary "CKM::lambda" \
    --vary "CKM::rhobar" \
    --vary "CKM::etabar" \
    --budget "FF" \
    --vary "B->K::fp_uncertainty@BZ2004v2" \
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
BINOPT=",model=SM,form-factors=BZ2004v2,l=mu,q=d"
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
    --vary "B->K::fp_uncertainty@BZ2004v2" \
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
BINOPT=",model=SM,form-factors=BZ2004v2,l=mu,q=u"
./src/clients/eos-evaluate \
    --parameter "mass::K0" 0.49368 \
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
    --vary "B->K::fp_uncertainty@BZ2004v2" \
    --budget "SL" \
    --vary "B->Pll::Lambda_pseudo@LargeRecoil" \
    --budget "C" \
    --vary "mass::c" \
    --budget "SD" \
    --vary "mass::b(MSbar)" \
    --vary "mass::t(pole)" \
    --vary "mu" \
    > ${PREFIX}bmtokmll-sm-binned-loq2.data


# Generate tarball
echo "Generating tarball"
tar zcf ${PREFIX%/}.tar.gz ${PREFIX}
