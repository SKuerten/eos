"""Compute Bayes factors between all combinations of models"""

from __future__ import division
import numpy as np
from numpy import exp, log

# use crop 200 everywhere

# log evidence for solution A(')
zposthep = {'I':560.982, 'II':571.304, 'III':560.424, 'sm':571.852}
zhpcqcd = {'I':565.70, 'II':580.676, 'III':569.629, 'sm':576.61}

# rescale prior volume a posteriori s.t. I-III on the same page
ratio = {'I':1/8**3, 'II':1/4**2, 'III':1/4**6, 'sm':1}

# naive Bayes factors
def bayes(z, ratio=None):
    for i in range(len(z.keys())):
        reference = z.keys()[i]
        zref = z[reference]
        if ratio:
            zref -= log(ratio[reference])
        for sc in z.keys()[i+1:]:

            zsc = z[sc]
            if ratio:
                zsc -= log(ratio[sc])
            factor = exp(zsc - zref)
            out_sc = sc
            out_ref = reference

            # prefer to put values > 1 in paper, then have to flip order
            if factor < 1:
                factor = 1. / factor
                out_sc = reference
                out_ref = sc

            print("%s vs %s: %g" % (out_sc, out_ref, factor))

print("Compute Bayes factors for...\n")
print("posthep13")
print("---------")
print("naive")
bayes(zposthep)

print
print("volume corrected")
bayes(zposthep, ratio)

print("\n\nposthep13hpqcd")
print("---------")
print("naive")
bayes(zhpcqcd)

print
print("volume corrected")
bayes(zhpcqcd, ratio)

print
print('NOTE: in each file, 200 highest weights are cropped.')

def ratios():
    '''scIII compare posterior masses. Use crop 200'''
    z = {'A':zhpcqcd['III'], 'B':569.175, 'C':567.448, 'D':568.439}

    zv = np.array([z[k] for k in sorted(z.keys())])
    zv = exp(zv)
    print('')
    print('Posterior mass ratios of the four solutions in scenario III hpqcd')
    print(sorted(z.keys()))
    print(zv / zv.sum())

ratios()
