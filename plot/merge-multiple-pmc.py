"""Merge the output of multiple PMC runs.

The idea is that each run covers one solution, and solutions do not
overlap. Hence the total evidence is just the sum of individual
evidences, and each self-normalized importance weight is reweighted by
N_a / N, where a is an index enumerating all solutions, so the number of
samples N_a in each run need not coincide.

"""

import plotScript
import h5py
import tables as pytables
import os
import numpy as np

def command_template(filename, crop=0):
    cmd_template = ''
    # input file
    cmd_template += ' %s' % filename
    cmd_template += ' --pmc-crop-outliers %d' % crop
    cmd_template += ' --pmc-queue-output'

    return cmd_template.split()

def merge(files, output):
    # read in all files
    margs = [plotScript.factory(command_template(f)) for f in files]

    # determine evidence
    Z = [m.integrate()[0] for m in margs]
    Z_total = sum(Z)

    # renormalize
    # print(len(margs[0].out.weights))
    nsamples = 0
    for i, m in enumerate(margs):
        nsamples += len(m.out.weights)

    for i, m in enumerate(margs):
        m.out.weights *= nsamples / len(m.out.weights)
        # transform to log scale for output in file
        m.out.weights = np.log(m.out.weights)

    # copy meta information from first file, assume it is equal for all
    merge_file = h5py.File(output, 'w')

    f = h5py.File(files[0], 'r')
    f.copy('/descriptions', merge_file)
    merge_file.create_group('/data')
    for g in ('/data/initial', '/data/statistics',):
        # merge_file.create_group(g)
        f.copy(g, merge_file['/data'])

    #copy data sets
    data_set_names = ('weights', 'broken', 'samples')
    for ds in data_set_names[0:2]:
        full_name = '/data/' + ds
        print('Creating ' + full_name)
        shape = [f[full_name].shape]
        shape[0] = nsamples
        shape = tuple(shape)
        print(shape)
        merge_file.create_dataset(full_name, dtype=f[full_name].dtype, shape=shape)

    # extra sausage for samples
    # h5py doesn't deal with the ndim vector in first element, so help it with the shapes
    ds = '/data/samples'
    print( f[ds][0].shape)
    samples = np.empty((nsamples, f[ds][0].shape[0]))
    merge_file.create_dataset(ds, data=samples)

    f.close()

    min_index = 0
    for i, fname in enumerate(files):
        with h5py.File(fname, 'r') as f:
            for ds in data_set_names[1:]:
                full_name = '/data/' + ds
                print('Copying ' + full_name)
                merge_file[full_name][min_index:min_index + f[full_name].len()] = f[full_name][:]

            merge_file['/data/weights'][min_index:min_index + f[full_name].len()] = margs[i].out.weights
            min_index += f[full_name].len()

if __name__ == '__main__':
    base = '/data/eos/2013-fall-erratum/2014-09-22/scIII_posthep13hpqcd'
    # pmc converged after different number of steps
    solution = ['A', 'B', 'C', 'D']
    files = [os.path.join(base, sol +'.hdf5')  for sol in solution]
    print(files)
    merge(files, output=os.path.join(base, 'pmc_multiple_merge.hdf5'))
