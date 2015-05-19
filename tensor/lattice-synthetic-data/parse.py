# python3 parse.py > out.log
from __future__ import print_function
import numpy as np
np.set_printoptions(precision=8, linewidth=120, threshold=50**2)

from matplotlib import pyplot as plt

def parse_form_factors(file_name, systematic=0.05):
    print('opening', file_name)
    with open(file_name) as f:
        lines = f.readlines()

        # number of parameters
        n = int(lines[0])
        mean = np.zeros(n)
        covariance = np.zeros((n, n))
        indices = {}

        # parse mean values
        for i, line in enumerate(lines[1:n+1]):
            name, value = line.split()
            indices[name] = i
            mean[i] = value

        # parse covariance
        for line in lines[n+1:]:
            name1, name2, value = line.split()
            covariance[indices[name1], indices[name2]] = value

        # symmetrize covariance
        covariance += covariance.transpose()
        covariance[np.diag_indices_from(covariance)] *= 0.5

        # systematic is relative to mean
        std_dev = systematic * mean

        # add systematic uncertainty
        covariance += np.outer(std_dev, std_dev)

        # standard deviation
        std_dev = np.sqrt(covariance.diagonal())

        # correlation matrix
        corr = covariance / np.outer(std_dev, std_dev)

        try:
            np.linalg.cholesky(covariance)
            print('Cholesky works')
        except np.LinalgError:
            raise

        return mean, covariance, corr, indices

def parse_qsquared(file_name):
    '''Parse (physical) q^2 values for each configuration'''
    res = {}
    with open(file_name) as f:
        for l in f:
            k, v = l.split()
            pos1 = k.find('_l')
            res[k[pos1+1:]] = v
    return res

def match_qsquared(indices, qsquared):
    '''Match q^2 for each FF value given configuration name.
    Return array with order of ``indices`` that contains the right q^2 value.'''
    res = np.zeros(len(indices))
    for k,v in indices.items():
        pos1 = k.find('_sl_l')
        res[v] = qsquared[k[pos1 + 4:]]
    return res

def kinematics(qsquared):
    s = '{{ '
    for i, qsq in enumerate(qsquared):
        if i > 0 and ((i % 3) == 0):
            s += '\n'
        s += 'Kinematics{{ "s", ' + str(qsq) + ' }}, '
    s += ' }},'
    return s

def eosify(arr):
    '''Return string representation easier for pasting into eos'''
    s = repr(arr)
    s = s.replace('array([', '{{')
    s = s.replace(')', '')
    s = s.replace('[ ', '{{')
    s = s.replace(']', '}}')
    return s

def plot_ff(tensor=False):
    tag = 't' if tensor else 'av'

    # parse data
    mean, cov, corr, indices = parse_form_factors('ff_phys_mass_' + tag + '_sl.d')
    std_dev = np.sqrt(cov.diagonal())
    qsquared_dict = parse_qsquared('qsqr_phys_mass_' + tag + '_sl.d')
    qsquared = match_qsquared(indices, qsquared_dict)

    labels = ['$T_1$', '$T_2$', '$T_{23}$'] if tensor else ['$V$', '$A_1$', '$A_{12}$', '$A_0$']
    delta_mF = [0.135, 0.55, 0.55] if tensor else [0.135, 0.550, 0.550, 0.087]
    npoints = len(qsquared) / len(labels)

    plt.clf()
    for i, l in enumerate(labels):
        slice = np.s_[i*npoints:(i+1)*npoints]
        # transform data with Blaschke factor
        blaschke =  1 / (1 - qsquared[slice] / (5.27958 + delta_mF[i])**2)
        mean[slice] *= blaschke
        std_dev[slice] *= blaschke
        plt.errorbar(qsquared[slice], mean[slice], yerr=std_dev[slice], fmt='o', label=l)

    plt.legend()
    plt.xlim(11, 19.21)
    plt.ylim(0, 2)
    plt.xlabel(r'$q^2$')
    plt.tight_layout()
    # plt.show()
    plt.savefig(tag + '.pdf')

    # output data
    print('q^2')
    print(kinematics(qsquared))
    print('mean')
    print(eosify(mean))
    print("std_dev with systematic uncertainty")
    print(repr(std_dev))
    print('relative uncertainty [%]')
    print(std_dev / mean * 100)

    # update covariance with Blaschke factor
    cov = np.outer(std_dev, std_dev) * corr
    print('covariance')
    print(eosify(cov))

plot_ff(tensor=False)
plot_ff(tensor=True)
