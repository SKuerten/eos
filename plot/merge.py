#! /usr/bin/python

import commands
import glob
import os
import shutil

# solution from http://stackoverflow.com/a/341730
def natsorted(strings):
    "Sort strings the way humans are said to expect."
    return sorted(strings, key=natural_sort_key)

def natural_sort_key(key):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', key)]

def merge_preruns(output_file_name, search='*.hdf5', input_files=None, strict=False, compression=False, use_pytables=False):
    """
    Merge a prerun, stored in different files into one common file.
    Before:
        file0:/prerun/chain #0/ file1:/prerun/chain #0 ...
    After:
        file0:/prerun/chain #0/ file0:/prerun/chain #1 ...
    """

    #extract working directory from output file
    base_dir, base_name = os.path.split(output_file_name)

    #find hdf5 files to merge
    if input_files is None:
        search_results = natsorted(glob.glob(os.path.join(base_dir, search)))
    else:
        search_results = input_files
        
    if len(search_results) == 0:
        raise Exception("No files matching '%s' in %s were found!" % (search, base_dir))

    #create merge file
    """
    cmd = '/afs/ipp/home/f/fdb/.local/bin/ptrepack'
    cmd += ' --complevel=%d' % (compression if compression is not None else 0,)
    cmd += ' -o %s %s' % (search_results[0], output_file_name)
    commands.getstatusoutput(cmd)
    """
    shutil.copy(search_results[0], output_file_name)

    groups = ['/prerun', '/descriptions/prerun']
    
    if use_pytables:
        import tables as pytables
        output_file = pytables.openFile(output_file_name, "r+")
        chains_per_file = len(output_file.listNodes(groups[0]))
    else:
        import h5py
        output_file = h5py.File(output_file_name, "r+")
        chains_per_file = len(output_file[groups[0]].keys())
    for f_i, file_name in enumerate(search_results[1:]):
        
        print("merging %s into %s" % (file_name, output_file_name))
        # check agreement
        if strict:
            input_file = pytables.openFile(file_name,'r')
            print(chains_per_file)
            assert(chains_per_file == len(input_file.listNodes(groups[1])))
            assert(len(input_file.getNode(groups[1] + '/chain #0/constraints')) == len(output_file.getNode(groups[1] + '/chain #0/constraints')))
            assert(len(input_file.getNode(groups[1] + '/chain #0/parameters')) == len(output_file.getNode(groups[1] + '/chain #0/parameters')))
            input_file.close()
        
        # copy data
        for g in groups:
            for i in range(chains_per_file):
                if use_pytables:
                    cmd = 'ptrepack'
                    cmd += ' --complevel=%d' % int(compression or 0)
                    cmd += ' %s:%s/chain\ #%d %s:%s/chain\ #%d' % (file_name, g, i, output_file_name, g, (f_i + 1) * chains_per_file + i)
                    print(cmd)
                    status, output = commands.getstatusoutput(cmd)
                    if status != 0:
                        raise Exception("ptrepack reported:\n %s" % output)
                else:
                    input_file = h5py.File(file_name, 'r')
                    input_file.copy(g + "/chain #%d" % i, output_file, name=g + "/chain #%d" % ((f_i + 1) * chains_per_file + i))
                    input_file.close()
                    
    print("Merged %d files together" % len(search_results))

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
    parser.add_argument('--descriptions', dest='desc',
                        help='Copy descriptions from this HDF5 input file to output', action='store')
    parser.add_argument('--search', dest='search',
                        help='HDF5 input file name pattern', action='store', default='*.hdf5')
    parser.add_argument('--output',
                        help='Output file name', action='store')
    parser.add_argument('--strict', dest='strict', default=False,
                        help='Check congruence', action='store_true')
    parser.add_argument('--compression', dest='compression', default=0,
                        help='Compress the copied data with a compression level. Default: 0', action='store')
    parser.add_argument('--input-file-list', dest='input_file_list',
                        help='List input files as text, one line per file in a file', action='store')
    parser.add_argument('--sm-unc', dest='sm_unc',
                        help='Merge uncertainty propagation files', action='store_true')
    
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
        
    if args.sm_unc:
        merge_sm_unc(output_file_name=args.output, input_files=input_files)
    else:
        # default: merge mcmc
        merge_preruns(output_file_name=args.output, search=args.search, input_files=input_files, strict=args.strict, compression=int(args.compression))
    
if __name__ == '__main__':
#    merge_preruns(search='scenario2*.hdf5')
    main()
