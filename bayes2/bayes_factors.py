"""Compute Bayes factors between all combinations of models"""

from __future__ import division, print_function
import numpy as np
from numpy import exp, log

# use crop 200 everywhere

# log evidence for solution A(')
zposthep = {'I':562.1, 'II':568.7, 'III':557.8, 'sm':572.3}
zhpcqcd  = {'I':570.1, 'II':577.6, 'III':566.372, 'sm':580.2}
zhpcqcdflat = {'II':580.708, 'III':573.749, 'sm':577.58}

# these ranges should be just large enough to contain solution A(')
desired_ranges = {'C7':0.4, 'C9':4, 'C10':4, "C7'":0.4, "C9'":4, "C10'":4}

# rescale prior volume a posteriori s.t. I-III on the same page
# ratio = {'I':1/8**3, 'II':1/3 * 1/4, 'III':1/4**5 * 1/3, 'sm':1}
# print(ratio)
actual_ranges_hpqcd = {'I': {'C7':4, 'C9':30, 'C10':30},
                       'II': {'C9':15, "C9'":15},
                       'III': {'C7':2, 'C9':15, 'C10':15, "C7'":2, "C9'":15, "C10'":15}}
actual_ranges_hpqcd_flat = {'II': {'C9':15, "C9'":15},
                            'III': {'C7':1, 'C9':7.5, 'C10':7.5, "C7'":0.8, "C9'":8, "C10'":6}}

def compute_ratios(actual_ranges):
    ratios = {}
    for scen, ranges in actual_ranges.iteritems():
        r = 1.
        for C in ranges.keys():
            r *= desired_ranges[C] / ranges[C]

        ratios[scen] = r
    return ratios

ratio = compute_ratios(actual_ranges_hpqcd)
ratio_flat = compute_ratios(actual_ranges_hpqcd_flat)

# add the sm, which is unscaled
for r in ratio, ratio_flat:
    r['sm'] = 1

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

print('desired ranges', desired_ranges)

print("posthep13")
print("---------")

print()
print("volume corrected")
bayes(zposthep, ratio)

print("\n\nposthep13hpqcd")
print("---------")

print()
print("volume corrected")
bayes(zhpcqcd, ratio)

print("\n\nposthep13hpqcdflat")
print("---------")
print("volume corrected")
bayes(zhpcqcdflat, ratio_flat)

print()
print('NOTE: in each file, 200 highest weights are cropped.')

def ratios():
    '''scIII compare posterior masses. Use crop 200'''
    z1 = {'A':zposthep['III'], 'B':556.8, 'C':556.9, 'D':557.7}
    z2 = {'A':zhpcqcd['III'], 'B':565.425, 'C':566.233, 'D':566.08}

    for z, name in zip((z1, z2), ('posthep13', 'posthep13hpqcd')):
        zv = np.array([z[k] for k in sorted(z.keys())])
        zv = exp(zv)
        print('')
        print('Posterior mass ratios of the four solutions in scenario III ' + name)
        print(sorted(z.keys()))
        print(zv / zv.sum())

ratios()
