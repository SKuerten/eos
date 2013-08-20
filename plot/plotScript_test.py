import os
from os.path import join
import tempfile
import shutil

import plotScript

"""
Acceptance test for plotScript

Sample files are created in a temporary folder
and deleted after the tests.

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

class TestPlotScript(object):
    base_name = '/tmp/nose'
    remove = False
    index = 0
    scen='scV'
    constraints='bsmumu'

    @classmethod
    def full_dir(cls):
        """Directory in which output files are stored"""
        return join(cls.base_name, '%s_%s' % (cls.scen, cls.constraints))

    @classmethod
    def setup_class(cls):
        print("Setting up test files...")
        # store in temporary directory
        if cls.base_name is None:
            cls.base_name = tempfile.mkdtemp()

        destination = 'BASE_NAME=%s' % cls.base_name

        # mcmc prerun
        run_command(form_command(destination, cls.scen, cls.constraints, action='pre', index=cls.index))

        # no need to merge
        link = join(cls.full_dir(), 'mcmc_pre_merged.hdf5')
        if not os.path.exists(link):
            os.symlink(join(cls.full_dir(), 'mcmc_pre_%d.hdf5' % cls.index), link)

        # pmc main run
        run_command(form_command(destination, scen='scV', constraints='bsmumu', action='pmc'))

    @classmethod
    def teardown_class(cls):
        print("Tearing down resources...")
        if cls.remove:
            if cls.base_name is not None and os.path.exists(cls.base_name):
                shutil.rmtree(cls.base_name)

    def test_pre(self):
        cmd = os.path.join(self.base_name, 'scV_bsmumu/mcmc_pre_%d.hdf5 ' % self.index)
        cmd += '--mcmc --pre  --single-1D 4'
        marg = plotScript.factory(cmd.split())
        marg.plot()

    def test_pmc(self):
        cmd = join(self.full_dir(), 'pmc_monolithic.hdf5')
        cmd += ' --nuis'
        marg = plotScript.factory(cmd.split())
        marg.plot()
