from __future__ import print_function
import numpy as np
np.set_printoptions(precision=10, linewidth=100, threshold=50**2)

def form_factors(file_name, systematic=0.05):
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

        # correlation matrix
        tmp = np.sqrt(covariance.diagonal())
        tmp = np.outer(tmp, tmp)
        corr = covariance / tmp

        # standard deviation
        std_dev = np.sqrt(covariance.diagonal())

        print('mean')
        print(repr(mean))
        print("std_dev with systematic uncertainty")
        print(repr(std_dev))
        print('relative uncertainty [%]')
        print(std_dev / mean * 100)
        print('correlation matrix')
        print(repr(corr))

        try:
            np.linalg.cholesky(covariance)
            print('Cholesky works')
        except np.LinalgError:
            print('Cholesky does NOT work')

form_factors('ff_phys_mass_av_sl.d')
form_factors('ff_phys_mass_t_sl.d')
