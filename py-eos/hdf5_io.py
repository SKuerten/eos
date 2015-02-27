'''Input/output to HDF5'''
from __future__ import print_function

import pypmc
import h5py
import numpy as np
import os, re, sys

sys.path.append(os.path.realpath('../plot'))
from samplingOutput import EOS_PYPMC_IS

def primary_group(step):
    '''Return directory name for given step (integer).'''
    if step is None:
        return ''
    return '/step #%d' % step

def save_analysis(file, directory, analysis):
    """Store analysis in human-readable format in hdf5 file.

    Example:
    $ h5ls -r file.hdf5
    /                        Group
    /descriptions            Group
    /descriptions/constraints Dataset {3}
    /descriptions/parameters Dataset {15}

    $ h5ls -d file.hdf5/descriptions/constraints
    constraints              Dataset {3}
    Data:
        (0) "B->K::f_0+f_++f_T@HPQCD-2013A",

    $ h5ls -d file.hdf5/descriptions/parameters
    parameters               Dataset {15}
    Data:
        (0) "Parameter: Re{cT}, prior type: flat, range: [-1,1], value = 0.981878",

    """

    desc  = 'descriptions'
    const = 'constraints'
    param = 'parameters'

    with h5py.File(file, 'a') as file:
        if desc in file[directory]:
            return

        group = file[directory].create_group(desc)

        # variable-length ASCII string
        dt = h5py.special_dtype(vlen=bytes)
        ds_const = group.create_dataset(const, (len(analysis.constraints),), dtype=dt)
        dt = np.dtype({'names': ['name', 'min', 'max', 'nuisance', 'info'],
                       'formats': [dt, np.float64, np.float64, np.uint8, dt]})
        ds_param = group.create_dataset(param, (len(analysis.priors),), dtype=dt)

        for i, c in enumerate(analysis.constraints):
            ds_const[i] = c.name

        ana_str = repr(analysis)

        # index where a line with parameter info starts
        line_start = ana_str.find('Parameter: ')
        i = 0
        while line_start != -1:
            end = ana_str.find(', value = ', line_start)
            line = ana_str[line_start:end]
            p = analysis.priors[i]
            ds_param[i] = (p.name, p.range_min, p.range_max, p.nuisance, line)
            i += 1
            # search forward in next line
            line_start = ana_str.find('Parameter: ', end)

        assert i == len(analysis.priors)

def exists_mixture(file, directory):
    '''Return whether mixture exists where it would be written to with ``save_mixture``.'''

    with h5py.File(file, 'r') as file:
        try:
            if 'covariances' in file[directory]:
                return True
        except KeyError:
            pass
    return False

def save_mixture(file, directory, mixture):
    '''Support Gauss and Students't mixtures'''
    with h5py.File(file, 'a') as file:
        means, covs, weights = pypmc.density.mixture.recover_gaussian_mixture(mixture)
        file.create_dataset(directory + '/means', data=means)
        file.create_dataset(directory + '/covariances', data=covs)
        file.create_dataset(directory + '/weights', data=weights)

        # student's t has extra data: dof
        try:
            dofs = [c.dof for c in mixture.components]
            file.create_dataset(directory + '/dofs', data=weights)
        except AttributeError:
            pass

def save_vb_hyperparameters(file, directory, vb):
    with h5py.File(file, 'a') as file:
        file.create_dataset(directory + '/alpha', data=vb.alpha)
        file.create_dataset(directory + '/beta', data=vb.beta)
        file.create_dataset(directory + '/nu', data=vb.nu)
        file.create_dataset(directory + '/m', data=vb.m)
        file.create_dataset(directory + '/W', data=vb.W)

def read_vb_hyperparameters(file, directory):
    kwargs = {}
    with h5py.File(file, 'r') as file:
        kwargs['alpha'] = file[directory + '/alpha'][:]
        kwargs['beta'] = file[directory + '/beta'][:]
        kwargs['nu'] = file[directory + '/nu'][:]
        kwargs['m'] = file[directory + '/m'][:]
        kwargs['W'] = file[directory + '/W'][:]

    return kwargs

def save_is_samples(file, directory, history_list):
    '''Assume each element in ``history_list`` has same length!'''
    with h5py.File(file, 'a') as file:
        # choose first run of first item in history_list
        chunk_size = len(history_list[0][0])
        n_samples = len(history_list) * chunk_size

        # 1st col: weights, other cols: parameter values
        dim = history_list[0].dim - 1
        weights_ds = file.create_dataset(directory + '/weights', (n_samples,), dtype=np.float64)
        samples_ds = file.create_dataset(directory + '/samples', (n_samples, dim), dtype=np.float64)
        for i, h in enumerate(history_list):
            weights_ds[i*chunk_size:(i+1)*chunk_size] = h[0][:,0]
            samples_ds[i*chunk_size:(i+1)*chunk_size] = h[0][:,1:]

def read_is_history(file, directory, last_step=None):
    '''Return ((list(samples), merged_samples), (list(weights), merged_weights), list(proposals))'''
    samples, weights, proposals = [], [], []
    with h5py.File(file, 'r') as f:
        # find all step directories
        groups = list(f[directory].keys())
        steps = sorted(filter(lambda x:re.search(r'^step', x), groups))
        # determine dimensions of samples
        N_total = 0
        for i, step in enumerate(steps[:last_step + 1]):
            ds = f[step]['importance_samples/samples']
            N_total += len(ds)
            dim = ds.shape[1]
        # create arrays big enough to hold all samples or weights
        histories = (pypmc.tools.History(dim, N_total), pypmc.tools.History(1, N_total))

        # read samples
        for i, step in enumerate(steps[:last_step + 1]):
            for x, l, h in zip(('samples', 'weights'), (samples, weights), histories):
                ds = f[step]['importance_samples/' + x]
                # extract individual runs as arrays (views, not copies!) in the list,
                # but h has the full story merged
                a = h.append(ds.len())
                # quirk: history always matrix-like, so change matrix with one column into ravelled array
                if a.shape[1] == 1:
                    a = a.view().reshape((len(a),))
                # don't make copy of array
                ds.read_direct(a)
                l.append(a)

    # can't open file multiple times, so stay away from context manager
    for i, step in enumerate(steps[:last_step + 1]):
        proposals.append(EOS_PYPMC_IS.read_mixture(file, step + '/vb'))
    return (samples, histories[0]), (weights, histories[1]), proposals

def save_combined_weights(file, directory, combined_weights):
    '''Save combined weights stored in the History separately for each step'''
    with h5py.File(file, 'a') as f:
#         last_step = len(combined_weights)
        grp = f.create_group(directory + '/combination') #%d' %  (last_step - 1))
        for i in range(len(combined_weights)):
            ds = grp.create_dataset('weights #%d' % i, data=combined_weights[i])
