import eos_scan_mc
from os import environ

def make_analysis(source):
    if source == "env":
        cmd_line = environ["EOS_CONSTRAINTS"] + environ["EOS_SCAN"] + environ["EOS_NUISANCE"]
        parser = eos_scan_mc.Parser(cmd_line)
        return eos_scan_mc.eos.Analysis(parser.constraints, parser.priors)
    else:
        module_hierarchy = source.split('.')
        module_to_import = str.join('.', module_hierarchy[:-1])
        analysis_name = module_hierarchy[-1]
        exec("from " + module_to_import + " import " + analysis_name + " as analysis")
        return analysis
