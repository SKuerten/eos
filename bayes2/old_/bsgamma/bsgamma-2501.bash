#!/bin/bash
#SBATCH --mail-type=all
#SBATCH --mail-user=vandyk@tp1.physik.uni-siegen.de
#SBATCH --mincpus=3
#SBATCH --nodes=1
#SBATCH --output=/scratch/vandyk/bayes2/2013-06-21/bsgamma-2501.out
#SBATCH --error=/scratch/vandyk/bayes2/2013-06-21/bsgamma-2501.err
#SBATCH --time=6:00:00
# vim: set sts=4 et:

DATADIR="/scratch/vandyk/bayes2/2013-06-21/"

mkdir -p ${DATADIR}
srun eos-scan-mc \
    --global-option model WilsonScan \
    --global-option scan-mode cartesian \
    --constraint "B->X_sgamma::BR[1.8]@BaBar-2012" \
    --constraint "B->X_sgamma::E_1[1.8]+E_2[1.8]@BaBar-2012" \
    --constraint "B->X_sgamma::E_1[1.8]+E_2[1.8]@Belle-2008" \
    --global-option form-factors KMPW2010 \
    --constraint "B^0->K^*0gamma::BR@CLEO-2000" \
    --constraint "B^+->K^*+gamma::BR@CLEO-2000" \
    --constraint "B^0->K^*0gamma::BR@BaBar-2009" \
    --constraint "B^+->K^*+gamma::BR@BaBar-2009" \
    --constraint "B^0->K^*0gamma::BR@Belle-2004" \
    --constraint "B^+->K^*+gamma::BR@Belle-2004" \
    --constraint "B^0->K^*0gamma::S_K+C_K@BaBar-2008" \
    --constraint "B^0->K^*0gamma::S_K+C_K@Belle-2006" \
    --scan "Re{c7}"   -0.6  +0.6  --prior flat \
    --scan "Re{c7'}"  -0.6  +0.6  --prior flat \
    --scan "mass::b(MSbar)" 4.10 4.5 --prior flat \
    --scan "B->K^*::F^V(0)@KMPW2010" 0.1 0.7 --prior flat \
    --scan "B->K^*::F^A1(0)@KMPW2010" 0.1 0.6 --prior flat \
    --chains 3 \
    --chunks 40 \
    --chunk-size 1000 \
    --seed 2501 \
    --output "${DATADIR}/bsgamma-2501.hdf5"
exit

