import plotScript
import samplingOutput

import os
from os.path import join
import shutil
import tempfile
import unittest

"""
Acceptance test for plotScript

Sample files are created in a temporary folder
and deleted after the tests.

Run with nosetests -v -s

"""

def run_command(cmd):
    """Run command in shell, return the output"""
    import commands

    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        raise Exception(output)

    return output

def form_command(dest, scen, constraints, action, index=0):
    cmd = dest + ' '
    cmd += os.path.join(os.environ['EOS_SCRIPT_PATH'], '%s-%s' % (scen, constraints)) + '.bash'
    cmd += ' %s %s' % (action, index)
    return cmd

class TestPlotScript(unittest.TestCase):
    base_name = '/tmp/nose'
    remove = False
    index = 0
    scen='scV'
    constraints='bsmumu'


    def full_dir(self):
        """Directory in which output files are stored"""
        return join(self.base_name, '%s_%s' % (self.scen, self.constraints))


    def setUp(self):
        print("Setting up test files...")
        # store in temporary directory
        if self.base_name is None:
            self.base_name = tempfile.mkdtemp()

        destination = 'BASE_NAME=%s' % self.base_name

        # mcmc prerun
        if not os.path.exists(self.mcmc_file_name()):
            run_command(form_command(destination, self.scen, self.constraints, action='pre', index=self.index))

        # no need to merge
        link = join(self.full_dir(), 'mcmc_pre_merged.hdf5')
        if not os.path.exists(link):
            os.symlink(join(self.full_dir(), 'mcmc_pre_%d.hdf5' % self.index), link)

        # pmc main run
        if not os.path.exists(self.pmc_file_name()):
            run_command(form_command(destination, self.scen, self.constraints, action='pmc'))


    def tearDown(self):
        print("Tearing down resources...")
        if self.remove:
            if self.base_name is not None and os.path.exists(self.base_name):
                shutil.rmtree(self.base_name)


    def mcmc_file_name(self):
        return os.path.join(self.base_name, 'scV_bsmumu/mcmc_pre_%d.hdf5' % self.index)


    def pmc_file_name(self):
        return os.path.join(self.base_name, 'scV_bsmumu/pmc_monolithic.hdf5')

    def mcmc_output(self, out):
        assert(len(out.samples) > 0)
        assert(len(out.samples[0]) > 0)
        assert(len(out.samples) == len(out.weights))
        # log posterior stored in last column
        self.assertEqual(len(out.samples[0]), len(out.par_defs) + 1)
        self.assertEqual(len(out.samples[0]), len(out.priors) + 1)

    def single_plot(self, cmd_line, mcmc=True, pmc=False):
        if pmc:
            cmd = self.pmc_file_name()
        else:
            cmd = self.mcmc_file_name()
        cmd += ' ' + cmd_line
        marg = plotScript.factory(cmd.split())
        marg.plot()

    def test_pmc(self):
        self.single_plot(' --single-1D 4', pmc=True)

    def test_pmc_proposal(self):
        self.single_plot('--pmc-proposal --single-2D 0 1 --nuisance', pmc=True)

    def test_pmc_integrate(self):
        self.skipTest("verify numbers and accuracy")
        cmd = self.pmc_file_name() + ' --integrate'
        marg = plotScript.factory(cmd.split())
        (integral, ratio, total_weight, error) = marg.integrate()
        self.assertEqual(ratio, 1)
        # value copied from file
        self.assertAlmostEqual(abs((integral - 1.3137666433807e16) / integral), 0, delta=1e-14)

    def test_pmc_component_integrate(self):
        self.skipTest("verify numbers and accuracy")
        cmd = self.pmc_file_name() + ' --integrate'
        marg = plotScript.factory(cmd.split())
        (weighted_average, weighted_std_dev), (rough_average, rough_std_dev) = marg.comp_integrate()

    def test_pmc_integrate_cuts(self):
        self.skipTest("verify numbers and accuracy")
        cmd = self.pmc_file_name() + ' --integrate --cut 0 MIN 0'
        marg = plotScript.factory(cmd.split())
        (integral, ratio, total_weight, error) = marg.integrate()
        self.assertAlmostEqual(ratio, 0.5, places=2)
        # value copied from file
        self.assertAlmostEqual(abs((integral - 0.5 * 1.3137666433807e16) / integral), 0, delta=1e-2)

    def test_mcmc_open(self):
        out = samplingOutput.MCMC_Output(self.mcmc_file_name())
        self.mcmc_output(out)

    def test_open(self):
#         self.skipTest("checking")
        op = samplingOutput.SamplingOutput.open
        self.assertRaises(IOError, op, self.mcmc_file_name() + 'asdfsd')
        self.mcmc_output(op(self.mcmc_file_name()))

    def test_pre_1D(self):
        self.single_plot(' --mcmc --pre  --single-1D 4')

    def test_pre_2D(self):
        self.single_plot(' --mcmc --pre --nuisance --single-2D 0 1')

    def test_pre_single_chain(self):
        self.single_plot(' --mcmc --pre  --single-1D 3 --chain 1')

    def test_pre_kde_1D(self):
        self.single_plot(' --mcmc --pre  --use-KDE --single-1D 3')

    def test_pre_kde_2D(self):
        self.single_plot(' --mcmc --pre  --use-KDE --nuisance --single-2D 0 1 --bandwidth 0.01')

    def test_pre_cont_1D(self):
        self.single_plot('--mcmc --pre --contours --single-1D 0')

    def test_pre_cont_2D(self):
        self.single_plot('--mcmc --pre --contours --nuisance --single-2D 0 1')

    def test_pre_kde_cont_1D(self):
        self.single_plot('--mcmc --pre --use-KDE --contours --single-1D 0')

    def test_pre_kde_cont_2D(self):
        self.single_plot('--mcmc --pre --use-KDE --contours --single-2D 0 1 --nuisance --bandwidth 0.01')

    def test_pre_multi_single_1D(self):
        self.single_plot('--mcmc --pre  --single-1D 1 3 --nuisance')
    def test_pre_multi_single_2D(self):
        self.single_plot('--mcmc --pre  --single-2D 1 3 --single-2D 1 4 --nuisance')
