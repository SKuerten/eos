import os
import tempfile
import shutil

def run_command(cmd):
    """Run command in shell, return the output"""
    import commands

    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        raise Exception(output)

    return output

class TestPlotScript(object):
    base_name = None
        
    @classmethod
    def setup_class(cls):
        print("Setting up test files...")
        # store in temporary directory
        cls.base_name = tempfile.mkdtemp()
        
        cmd = 'BASE_NAME=%s ' % cls.base_name
        
        # mcmc prerun
        cmd += os.path.join(os.environ['EOS_SCRIPT_PATH'], 'scV-bsmumu.bash')
        cmd += ' pre 1'
        print(cmd)
        run_command(cmd)
        
    @classmethod
    def teardown_class(cls):
        print("Tearing down resources...")
        if cls.base_name is not None and os.path.exists(cls.base_name):
            shutil.rmtree(cls.base_name)
    
    def test_harr(self):
        assert 1