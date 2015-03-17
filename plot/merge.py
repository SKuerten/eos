#! /usr/bin/env python
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
                cut_off=None):
    '''
    Merge Markov chains from eos-to-pypmc interface.
    '''
    if not input_files:
        input_files = search_files(search, output_file_name)

    # hdf5 groups
    groups = ['/samples', '/descriptions', '/log_posterior']

    f = h5py.File(input_files[0], 'r')
    par = f['/chain #0/descriptions/parameters'][:]
    constraints = f['/chain #0/descriptions/constraints'][:]
    n_samples = len(f['/chain #0/samples'])
    f.close()

    # count chains copied
    nchains_copied = 0

    output_file = h5py.File(output_file_name, "w")
    for g in groups:
        output_file.create_group(g)

    for f_i, file_name in enumerate(input_files):
        print("merging %s" % file_name)
        input_file = h5py.File(file_name, 'r')
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
                input_file.copy('/chain #%d' % i + g, output_file, name='/chain #%d' % nchains_copied + g)
            nchains_copied += 1
        input_file.close()

    print("Merged %d chains of %d files" % (nchains_copied, len(input_files)))

    output_file.close()

def merge_sm_unc(output_file_name, input_files):
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

    parser  = argparse.ArgumentParser(description='Merge MCMC input files')
    parser.add_argument('--compression', dest='compression', default=0,
                        help='Compress the copied data with a compression level. Default: 0', action='store')
    parser.add_argument('--cut-off', help='Skip chains whose maximum posterior is below the cut-off', action='store', default=None)
    parser.add_argument('--descriptions', dest='desc',
                        help='Copy descriptions from this HDF5 input file to output', action='store')
    parser.add_argument('--input-file-list', dest='input_file_list',
                        help='List input files as text, one line per file in a file', action='store')
    parser.add_argument('--output',
                        help='Output file name', action='store')
    parser.add_argument('--search', dest='search',
                        help='HDF5 input file name pattern', action='store', default='mcmc_*.hdf5')
    parser.add_argument('--sm-unc', dest='sm_unc',
                        help='Merge uncertainty propagation files', action='store_true')
    parser.add_argument('--pypmc',
                        help='Merge mcmc from pypmc', action='store_true')

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.getcwd(), 'mcmc_pre_merged.hdf5')

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

    if args.sm_unc:
        merge_sm_unc(output_file_name=args.output, input_files=input_files)
    elif args.pypmc:
        merge_pypmc(output_file_name=args.output, search=args.search,
                      input_files=input_files, cut_off=cut_off)
    else:
        # default: merge mcmc
        merge_preruns(output_file_name=args.output, search=args.search,
                      input_files=input_files,
                      compression=int(args.compression), cut_off=cut_off)

if __name__ == '__main__':
#    merge_preruns(search='scenario2*.hdf5')
    main()
