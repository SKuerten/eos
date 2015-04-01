#! /usr/bin/env python

'''Uncertainty propagation with `eos-evaluate`.'''

from __future__ import print_function, division

import argparse
import h5py
import numpy as np
import commands, os, sys

sys.path.append(os.path.realpath('../plot'))
import samplingOutput

hdf5_subdirectory = '/unc'
class Unc(object):
    def __init__(self, args):
        self.args = args

        # read MCMC input file
        select = [int(x) for x in args.input_range]
        self.input = samplingOutput.EOS_PYPMC_MCMC(self.args.mcmc_input, skip_initial=0.0,
                                                   select=select)

        self.fixed_name = args.fix[0]

        self.results = np.empty((len(self.input.samples), int(args.fix[-1])))

        desc = 'descriptions'
        with h5py.File(args.output, 'w') as output_file:
            with h5py.File(args.mcmc_input, 'r') as input_file:
                input_file.copy('/chain #0/' + desc, output_file, name=desc)

            ds_par = output_file[desc + '/parameters']
            ds_par.attrs['input file'] = self.args.mcmc_input
            ds_par.attrs['min sample index'] = select[0]
            ds_par.attrs['max sample index'] = select[1]

    def run(self):
        for n, x in enumerate(self.input.samples):
            cmd = self.build_cmd(x)
            status, output = commands.getstatusoutput(cmd)
            if status:
                print(cmd)
                print(output)
                exit(status)
            # ignore first two lines, then extract second column
            arr = np.loadtxt(output.splitlines()[2:])
            self.results[n] = arr.T[1]

        # dump results
        with h5py.File(args.output, 'a') as output_file:
            ds_obs = output_file.create_dataset('/observable', data=self.results)
            ds_obs.attrs['name'] = self.args.observable

            # fixed values always the same, only use from last output
            ds_fixed = output_file.create_dataset('/descriptions/fixed parameter', data=arr.T[0])
            ds_fixed.attrs['name'] = self.fixed_name

    def build_cmd(self, x):
        cmd = 'eos-evaluate --precision 16'
        if args.eos_integration_points:
            cmd += ' --integration-points %d' % args.eos_integration_points
        for i, p in enumerate(self.input.par_defs):
            cmd += ' --parameter "' + p.name + '" %.16f' % x[i]
        cmd += ' --parameter-range %s %s %s %s' % tuple(self.args.fix)
        for k,v in self.args.kinematics.iteritems():
            cmd += ' --kinematics %s %s' % (k, v)
        for k,v in self.args.parameter.iteritems():
            cmd += ' --parameter %s %s' % (k, v)
        cmd += ' --observable "' + self.args.observable + '"'
        return cmd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--eos-integration-points", type=int)
    parser.add_argument('--fix', nargs=4,
                        help='EOS parameter name, min, max, npoints. Includes end point!. Example: "Re{cT} -1 1 11')
    parser.add_argument("--input-range", help='The range of input samples used. Example: --input-range 3800 4100', nargs=2)
    parser.add_argument("--kinematics", nargs='*',
                        help='''Use it to define kinematics.
                        Example: --kinematics s_min 1 s_max 6''')
    parser.add_argument("--mcmc-input", help='File name with MCMC samples. Only `chain #0` is used!')
    parser.add_argument('--observable',
                        help='''Specify observable for `eos-evaluate`. Watch the string quoting!
                        Example: '--observable "B->Kll::dBR/ds@LowRecoil,l=tau" ''')
    parser.add_argument("--output", const='', nargs='?',
                        help="Output file name.")
    parser.add_argument("--parameter", nargs='*',
                        help='''Set EOS parameter name(s) to value(s).
                        Example: --parameter mass::c 1.21 mass::b(MSBAR) 4.123''')

    args = parser.parse_args()

    ###
    # validate and use arguments
    ###
    # keyword and value stored after one another, can have more than one pair
    for key in ['kinematics', 'parameter']:
        arg = args.__dict__[key]
        nargs = len(arg)
        if nargs % 2 == 1:
            raise argparse.ArgumentError("Supply %s as 'key1 value1 key2 value2'" % key)
        d = dict()
        for i in range(nargs // 2):
            d[arg[2 * i]] = arg[2 * i + 1]
        args.__dict__[key] = d

    ###
    # take action
    ###
    unc = Unc(args)
    unc.run()
