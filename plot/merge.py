#! /usr/bin/env python
'''Merge multiple input files into one output file.'''

from __future__ import print_function
import commands
import glob
import h5py
import os
import shutil

import numpy as np

# solution from http://stackoverflow.com/a/341730
def natsorted(strings):
    "Sort strings the way humans are said to expect."
    return sorted(strings, key=natural_sort_key)

def natural_sort_key(key):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', key)]

def invalid_chain(file, chain, cut_off=None, format='eos'):
    """Check if mode larger than cut off"""
    if format == 'eos':
        # last value of the last mode is the best found in all updates sequences
        mode = file['/prerun/chain #%d/stats/mode' % chain][-1][-1]
    elif format == 'pypmc':
        mode = file['/chain #%d' % chain + '/log_posterior'][:].max()

    if cut_off is None:
        print('chain %d has mode %g' % (chain, mode))
        return False

    if mode > cut_off:
        return False
    else:
        print('skipping chain %d with mode %g < %g' % (chain, mode, cut_off))
        return True

def search_files(search, output_file_name):
    '''Search files matching pattern in directory of output file. Ignore the latter.'''
    #extract working directory from output file
    base_dir, base_name = os.path.split(output_file_name)

    #find hdf5 files to merge
    search_results = natsorted(glob.glob(os.path.join(base_dir, search)))
    # avoid double merging
    try:
        search_results.remove(output_file_name)
        print("Removed %s from input files" % output_file_name)
    except ValueError:
        pass

    if len(search_results) == 0:
        raise Exception("No files matching '%s' in %s were found!" % (search, base_dir))

    return search_results

def merge_preruns(output_file_name, search='*.hdf5', input_files=None,
                  compression=False, use_pytables=False, cut_off=None):
    """
    Merge a prerun, stored in different files into one common file.
    Before:
        file0:/prerun/chain #0/ file1:/prerun/chain #0 ...
    After:
        file0:/prerun/chain #0/ file0:/prerun/chain #1 ...
    """

    if not input_files:
        input_files = search_files(search, output_file_name)

    # hdf5 groups
    groups = ['/prerun', '/descriptions/prerun']

    # count chains copied
    nchains_copied = 0

    output_file = h5py.File(output_file_name, "w")
    for g in groups:
        output_file.create_group(g)

    for f_i, file_name in enumerate(input_files):
        print("merging %s" % file_name)
        input_file = h5py.File(file_name, 'r')
        nchains_in_file = len(input_file[groups[0]].keys())

        # copy data
        for i in range(nchains_in_file):
            if invalid_chain(input_file, i, cut_off, format='eos'):
                continue
            for g in groups:
                input_file.copy(g + "/chain #%d" % i, output_file, name=g + "/chain #%d" % nchains_copied)
            nchains_copied += 1
        input_file.close()

    print("Merged %d chains of %d files" % (nchains_copied, len(input_files)))

    output_file.close()

def merge_pypmc(output_file_name, search='mcmc_*.hdf5', input_files=None,
                cut_off=None, single_chain=True, skip_initial=0.0, thin=1):
    '''
    Merge Markov chains from eos-to-pypmc interface.
    '''
    if not input_files:
        input_files = search_files(search, output_file_name)

    # hdf5 groups
    groups = ['/samples', '/descriptions', '/log_posterior']
    first = '/chain #0'

    with h5py.File(input_files[0], 'r') as f:
        par = f[first + '/descriptions/parameters'][:]
        constraints = f[first + '/descriptions/constraints'][:]
        n_samples = len(f[first + '/samples'])

    # count chains copied
    nchains_valid = 0
    valid_files = []

    with h5py.File(output_file_name, "w") as output_file:
        for file_name in input_files:
            print("merging %s" % file_name)
            with h5py.File(file_name, 'r') as input_file:
                nchains_in_file = len(input_file['/'].keys())

                for i in range(nchains_in_file):
                    # check agreement
                    np.testing.assert_equal(input_file['/chain #%d/descriptions/parameters' % i][:], par)
                    np.testing.assert_equal(input_file['/chain #%d/descriptions/constraints' % i][:], constraints)
                    assert input_file['/chain #%d/samples' % i].len() == n_samples, 'expected %d samples, got %d samples for chain %d in %s' % (n_samples, input_file['/chain #%d/samples' % i].len(), i, file_name)

                    if invalid_chain(input_file, i, cut_off, format='pypmc'):
                        continue

                    # copy data
                    for g in groups:
                        if not single_chain:
                            input_file.copy('/chain #%d' % i + g, output_file, name='/chain #%d' % nchains_valid + g)
                    nchains_valid += 1
                valid_files.append(file_name)

        # compress into one big chain
        if single_chain:
            # with thinning and skipping, read fewer samples from each chain
            first_index = int(skip_initial * n_samples)
            new_samples = (n_samples - first_index) // thin
            n_samples_total = new_samples * nchains_valid
            print("Select %d samples from each chain" % new_samples)

            # predefine data sets large enough to hold everything
            grp = output_file.create_group(first)
            data_groups = [('/samples', (n_samples_total, len(par))), ('/log_posterior', (n_samples_total,))]
            data_sets = [output_file.create_dataset(first + g[0], g[1]) for g in data_groups]

            # loop over valid input files
            nchain = 0
            for j, file_name in enumerate(valid_files):
                with h5py.File(file_name, 'r') as input_file:
                    # copy descriptions only once
                    if j == 0:
                        s = first + '/descriptions'
                        input_file.copy(s, output_file, name=s)
                    # loop over chains
                    for i, c in enumerate(input_file['/'].iterkeys()):
                        for g, ds in zip(data_groups, data_sets):
                            old_length = nchain * new_samples
                            data = input_file[c + g[0]][first_index::thin]

                            # 1D versus 2D assignment
                            if g[0] == '/log_posterior':
                                ds[old_length:old_length + new_samples] = data
                            else:
                                ds[old_length:old_length + new_samples, :] = data
                    nchain += 1

    print("Merged %d chains from %d files" % (nchains_valid, len(input_files)))

def merge_unc_pypmc(output_file_name, search='unc*.hdf5', input_files=None):
    if not input_files:
        input_files = search_files(search, output_file_name)


    with h5py.File(output_file_name, "w") as output_file:
        # check if files agree with first input file
        with h5py.File(input_files[0], 'r') as f:
            constraints = f['/descriptions/constraints'][:]

            par_ds = f['/descriptions/parameters']
            parameters = par_ds[:]
            sample_input_file = par_ds.attrs['input file']

            fixed_par_ds = f['/descriptions/fixed parameter']
            fixed_par = fixed_par_ds[:]
            fixed_par_name = fixed_par_ds.attrs['name']

            obs_ds = f['observable']
            obs_name = obs_ds.attrs['name']

            print("Definining data format from %s" % os.path.split(input_files[0])[1])

            # copy meta data to output file
            f.copy('/descriptions', output_file)

            # create observable data set and initialize with first input file
            obs_ds = output_file.create_dataset('/observable', data=obs_ds[:], maxshape=(None, obs_ds.shape[1]))
            obs_ds.attrs['name'] = obs_name

        for file_name in input_files[1:]:
            with h5py.File(file_name, 'r') as input_file:
                # check agreement
                np.testing.assert_equal(input_file['/descriptions/constraints'][:], constraints)

                par_ds = input_file['/descriptions/parameters']
                np.testing.assert_equal(par_ds[:], parameters)
                assert par_ds.attrs['input file'] == sample_input_file

                fixed_par_ds = input_file['/descriptions/fixed parameter']
                np.testing.assert_equal(fixed_par_ds[:], fixed_par)
                assert fixed_par_ds.attrs['name'] == fixed_par_name

                in_obs_ds = input_file['/observable']
                assert in_obs_ds.attrs['name'] == obs_name

                print("merging %s" % os.path.split(file_name)[1])
                old_length = obs_ds.len()
                obs_ds.resize(old_length + in_obs_ds.len(), axis=0)
                obs_ds[old_length:old_length + in_obs_ds.len()] = in_obs_ds[:]

                # reset max. There may be holes in the range of min to max
                s = 'max sample index'
                par_ds.attrs[s]
                output_file['/descriptions/parameters'].attrs[s] = par_ds.attrs[s]

def merge_unc(output_file_name, input_files):
    """

    """
    import tables as pytables

    print("merging %s into %s" % (input_files[0], output_file_name))
    shutil.copy(input_files[0], output_file_name)
    optional_groups = ['/data/parameters']
    groups = ['/data/observables']

    output_file = pytables.openFile(output_file_name, "r+")
    n_obs = len(output_file.getNode('/descriptions/observables')._f_listNodes())
    n_par = len(output_file.getNode('/descriptions/parameters')._f_listNodes())
    n_samples = len(output_file.getNode('/data/observables'))


    for f_i, file_name in enumerate(input_files[1:]):
        print("merging %s into %s" % (file_name, output_file_name))

        input_file = pytables.openFile(file_name, 'r')

        # check agreement
        assert(n_obs == len(input_file.getNode('/descriptions/observables')._f_listNodes()))
        assert(n_par == len(input_file.getNode('/descriptions/parameters')._f_listNodes()))
        assert(n_samples == len(input_file.getNode('/data/observables')))

        # copy/append data
        for g in groups:
            output_file.getNode(g).append(input_file.getNode(g).read())

        for g in optional_groups:
           try:
               output_file.getNode(g).append(input_file.getNode(g).read())
           except pytables.exceptions.NoSuchNodeError:
               pass

        input_file.close()
    output_file.close()
    return

    # h5py doesn't work with nested arrays, see http://code.google.com/p/h5py/issues/detail?id=211
    """
    out_ds = output_file[g]
    out_ds.resize((n + input_file[g].len(),))
    for i, rec in enumerate(input_file[g]):
        #output_file[g][-input_file[g].len():] = input_file[g][:]
        print(rec)
        out_ds[n + i] = rec
    """

def main():

    import argparse

    parser  = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compression', dest='compression', default=0,
                        help='Compress the copied data with a compression level. Default: 0')
    parser.add_argument('--cut-off', default=None,
                        help='Skip chains whose maximum posterior is below the cut-off')
    parser.add_argument('--descriptions', dest='desc',
                        help='Copy descriptions from this HDF5 input file to output')
    parser.add_argument('--input-file-list', dest='input_file_list',
                        help='List input files as text, one line per file in a file')
    parser.add_argument('--output', help='Output file name')
    parser.add_argument('--pypmc', action='store_true',
                        help='Merge mcmc from pypmc')
    parser.add_argument('--search', default='mcmc_*.hdf5',
                        help='HDF5 input file name pattern')
    parser.add_argument('--single-chain', action='store_true',
                        help='Merge all input chains into one big output chain.')
    parser.add_argument('--skip-initial', type=float, default=0.,
                        help="Allows to skip the first fraction of iterations. Applies only with `--pypmc` and `--single-output`")
    parser.add_argument('--unc', action='store_true',
                        help='Merge uncertainty propagation files')
    parser.add_argument('--thin', type=int, default=1,
                        help='Thin MCMC samples. Applies only with `--pypmc` and `--single-output`')

    args = parser.parse_args()

    if args.output is None:
        if args.unc:
            output = 'unc_merged.hdf5'
        else:
            output = 'mcmc_pre_merged.hdf5'
        args.output = os.path.join(os.getcwd(), output)

    if args.desc is not None:
        import h5py

        input_file = h5py.File(args.desc, 'r')
        output_file = h5py.File(args.output, 'r+')
        try:
            pass
#            output_file.create_group('/descriptions')
        except ValueError:
            pass
        input_file.copy('/descriptions', output_file['/'])
        return

    print("Merging into output file %s" % args.output)

    input_files = None
    if args.input_file_list is not None:
        f = open(args.input_file_list, 'r')
        #remove trailing newline
        input_files = [name[:-1] for name in f.readlines()]

    if args.cut_off is not None:
        cut_off = float(args.cut_off)
    else:
        cut_off = None

    if args.unc:
        if args.pypmc:
            merge_unc_pypmc(output_file_name=args.output, input_files=input_files)
        else:
            merge_unc(output_file_name=args.output, input_files=input_files)
    else:
        if args.pypmc:
            merge_pypmc(output_file_name=args.output, search=args.search,
                        input_files=input_files, cut_off=cut_off,
                        single_chain=args.single_chain, skip_initial=args.skip_initial,
                        thin=args.thin)
        else:
            # default: merge mcmc
            merge_preruns(output_file_name=args.output, search=args.search,
                          input_files=input_files,
                          compression=int(args.compression), cut_off=cut_off)

if __name__ == '__main__':
    main()
