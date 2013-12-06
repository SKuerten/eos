"""Compute Bayes factors between all combinations of models"""

from __future__ import division
from numpy import exp, log

# log evidence
zposthep = {'I':560.99, 'II':571.32, 'III':560.45, 'sm':571.86}
zhpcqcd = {'I':565.71, 'II':580.74, 'III':569.49, 'sm':576.62}

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
            print("%s vs %s: %g" % (sc, reference, factor))

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
