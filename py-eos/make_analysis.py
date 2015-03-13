import eos_scan_mc
from os import environ

def make_analysis(source):
    if source == "env":
        cmd_line = environ["EOS_SCAN"] + environ["EOS_NUISANCE"] + environ["EOS_CONSTRAINTS"]
        parser = eos_scan_mc.Parser(cmd_line)
        try:
            return eos_scan_mc.eos.Analysis(parser.constraints, parser.priors)
        except eos_scan_mc.eos.EOSError as e:
            print("Caught EOSError when creating Analysis with \n constraints:\n" +
                  str(parser.constraints) + "\n\n and priors:\n" + str(parser.priors))
            raise
    else:
        module_hierarchy = source.split('.')
        module_to_import = str.join('.', module_hierarchy[:-1])
        analysis_name = module_hierarchy[-1]
        exec("from " + module_to_import + " import " + analysis_name + " as analysis")
        return analysis
