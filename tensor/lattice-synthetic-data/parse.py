import numpy as np

with open('ff_phys_mass_av_sl.d') as f:
    lines = f.readlines()
    n = int(lines[0])
    mean = np.zeros(n)
    covariance = np.zeros((n,n))
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

    

    print(mean)
    print(indices)
    print(covariance[0])
    print(covariance[1])

