"""Merge the output of multiple PMC runs.

The idea is that each run covers one solution, and solutions do not
overlap. Hence the total evidence is just the sum of individual
evidences, and each self-normalized importance weight is reweighted by
Z_a / Z, where a is an index enumerating all solutions. The number of
samples in each run need not coincide.

"""

import plotScript
import h5py
import tables as pytables
import os
import numpy as np

def command_template(filename, crop=100):
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
        R = Z[i] / Z_total
        print('ratio: %g' % R)
        m.out.weights /= m.out.weights.sum()
        m.out.weights *= R
        # transform to log scale for output in file
        m.out.weights = np.log(m.out.weights)
        nsamples += len(m.out.weights)

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

def merge_old(files, output):
    # read in all files
    margs = [plotScript.factory(command_template(f)) for f in files]

    # get samples and weights
    print(len(margs[0].out.samples))

    # determine evidence
    Z = [m.integrate()[0] for m in margs]
    Z_total = sum(Z)

    # renormalize
    # print(len(margs[0].out.weights))
    nsamples = 0
    for i, m in enumerate(margs):
        m.out.weights *= Z[i] / Z_total / m.out.weights.sum()
        nsamples += len(m.out.weights)

    # copy meta information from first file, assume it is equal for all
    merge_file = h5py.File(output, 'w')

    f = h5py.File(files[0], 'r')
    f.copy('/descriptions', merge_file)
    merge_file.create_group('/data')
    for g in ('/data/initial', '/data/statistics',):
        # merge_file.create_group(g)
        f.copy(g, merge_file['/data'])
    f.close()

        # create new data set with right dimensions to hold all samples
        # for g in data_groups:
        #     shape = f[g].shape
        #     if len(shape) == 1:
        #         shape = (nsamples,)
        #     else:
        #         shape = (nsamples, shape[1])
        #     print('%s: %s' % (g, shape))
        #     merge_file.create_dataset(g, dtype=f[g].dtype, shape=shape)

    # cant' get pytables to work with simple array
    # need extra sausage
    broken = np.zeros(nsamples, dtype=np.int8)
    min_index = 0
    for fname in files:
        with h5py.File(fname, 'r') as f:
            data_set = f['/data/broken']
            broken[min_index:min_index + data_set.len()] = data_set[:]

    merge_file['/data'].create_dataset('broken', data=broken)

    merge_file.close()


    # now use pytables for writing samples and weights with append
    merge_file = pytables.openFile(output, 'r+')

    # initial file
    f = pytables.openFile(files[0], 'r')

    data_groups = ('weights', 'samples')


    # dsets = [merge_file.createTable('/data', g, f.getNode('/data/' + g).read())       for g in data_groups[0:1]]
    # regular arrays need dtype
    # weights is a structured array, and is handled differently
    g = data_groups[-1]
    data_sets = [merge_file.createTable('/data', g, f.getNode('/data/' + g).read())]

    # arr = f.getNode('/data/broken').read()
    # data_sets.append(merge_file.createEArray('/data', 'broken', pytables.Atom.from_dtype(arr.dtype), (0,) ))
    # print(arr.dtype)
    # data_sets[-1].append(arr[:])

    arr = f.getNode('/data/samples').read()
    data_sets.append(merge_file.createEArray('/data', 'samples', pytables.Atom.from_dtype(arr.dtype), (0, arr.shape[1]) ))
    data_sets[-1].append(arr)

    # for g in data_groups[1:2]:
    #     print(f.getNode('/data/' + g).read().dtype)
    #     arr = f.getNode('/data/' + g).read()
    #     data_sets.append(merge_file.createEArray('/data', g, pytables.Atom.from_dtype(arr.dtype), (0, arr.shape[1]) ))
    f.close()

    for fname in files[1:]:
        f = pytables.openFile(fname, 'r')
        for i,g in enumerate(data_groups):
            data_sets[i].append(f.getNode('/data/' + g).read())
        f.close()

    """
    # store new samples and weights
    min_index = 0
    for i, m in enumerate(margs):
        n_i = len(m.out.weights)
        with h5py.File(files[i], 'r') as f:
            for g in data_groups[:-1]:
                print('merge file: %s: %s' % (g, merge_file[g].shape))
                print('input file: %s: %s' % (g, f[g].shape))
                merge_file[g][min_index:min_index + n_i] = m.out.samples[:]

            merge_file['/data/weights'][min_index:min_index + n_i, :] = m.out.weigths[:]
        min_index += n_i
    """
    merge_file.close()

if __name__ == '__main__':
    base = '/data/eos/2013-fall/2013-12-06/scIII_posthep13hpqcd-'
    # pmc converged after different number of steps
    nsteps = [4, 8, 5, 6]
    files = [os.path.join(base + str(i), 'pmc_parameter_samples_%d.hdf5_merge' % nsteps[i])  for i in range(len(nsteps))]
    print(files)
    merge(files, output='pmc_multiple_merge.hdf5')
