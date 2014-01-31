# vim: set sts=4 et :

import os, commands
from os import environ
import numpy as np
import sys
import time

## {{{ http://code.activestate.com/recipes/65222/ (r1)
import threading

class TaskThread(threading.Thread):
    """Thread that executes a task every N seconds"""

    def __init__(self):
        threading.Thread.__init__(self)
        self._finished = threading.Event()
        self._interval = 15.0
        self.return_value = None

    def setInterval(self, interval):
        """Set the number of seconds we sleep between executing our task"""
        self._interval = interval

    def shutdown(self, return_value=None):
        """Stop this thread"""
        self._finished.set()
        self.return_value = return_value

    def run(self):
        while 1:
            if self._finished.isSet():
                return self.return_value

            # sleep for interval or until shutdown
            self._finished.wait(self._interval)

            self.task()

    def task(self):
        """The task done by this thread - override in subclasses"""
        raise Exception("TaskThread.task not implemented")
## end of http://code.activestate.com/recipes/65222/ }}}

class JobInfo(object):
    def __init__(self, min, max, file_name):
        self.min = min
        self.max = max
        self.file_name = file_name
        self.script_name = ''
        self.broken = False

    def __repr__(self):
        return "min = %d, max = %d, file = %s, job script = %s" % \
               (self.min, self.max, self.file_name, self.script_name)

def env(variable, default=None):
    " Read environment variable. If no default value given, an exception is raised"
    try:
        return environ[variable]
    except KeyError:
        if default is None:
            raise
        else:
            return default

class PMC_Iterator(object):
    def __init__(self):
        """
        Read from environment variables
        """

        # current step of PMC
        self.step = 0

        ###
        # required environment variables
        ###
        self.output_base = environ['PMC_OUTPUT_BASE_NAME']

        # default: put log and job files in same directory as HDF5 output
        self.output_job_base = self.output_base

        ###
        # job manager options
        ###
        self.verbose = False
        self.clean_up = True

        ###
        # optional environment variables
        ###
        self.initialization_mode = env('PMC_INITIALIZATION_MODE', 'HierarchicalClustering')

        # always passed along, use it for --debug etc
        self.general_options = env('PMC_GENERAL_OPTIONS', "")
        self.convergence_options = env('PMC_CONVERGENCE', "")
        self.initialization_options = env('PMC_INITIALIZATION')
        self.ignore_groups = env('PMC_IGNORE_GROUPS','')
        self.group_by_r_value = float(env('PMC_GROUP_BY_RVALUE', 1.5))
        self.adjust_sample_size = int(env('PMC_ADJUST_SAMPLE_SIZE', 1))

        max_number_of_jobs = 5000
        try:
            self.number_of_jobs = min(int(environ['PMC_NUMBER_OF_JOBS']), max_number_of_jobs)
        except KeyError:
            self.number_of_jobs = max_number_of_jobs

        # shouldn't use multiple cores on a cluster
        #        self.thread_parallelization = int(env('PMC_PARALLEL', False))
        self.thread_parallelization = int(False)
        self.polling_interval = int(env('PMC_POLLING_INTERVAL', 30))

        # initialize from merged prerun directly
        if self.initialization_mode == 'HierarchicalClustering':
            self.prerun_file_name = env('PMC_MERGE_FILE')
            self.target_number_of_clusters = int(environ['PMC_CLUSTERS'])
            self.patch_length = int(environ['PMC_PATCH_LENGTH'])
            self.skip_initial = float(environ['PMC_SKIP_INITIAL'])

        elif self.initialization_mode == 'UncertaintyPropagation':
            self.uncertainty_analysis = environ['PMC_UNCERTAINTY_ANALYSIS']
            self.uncertainty_input = environ['PMC_UNCERTAINTY_INPUT']
            self.uncertainty_sample_directory = env('PMC_UNCERTAINTY_SAMPLE_DIRECTORY', '')

            # determine default
            if not self.uncertainty_sample_directory:
                import h5py
                f = h5py.File(self.uncertainty_input)
                allowed_dirs = ('/data', '/data/final')
                for dir in allowed_dirs:
                    try:
                        len(f[dir + '/samples'])
                        self.uncertainty_sample_directory = dir
                        break
                    except KeyError:
                        continue
            if not self.uncertainty_sample_directory:
                raise Exception("Cannot find the samples directory in %s. Please indicate it via PMC_UNCERTAINTY_SAMPLE_DIRECTORY" % str(allowed_dirs))
        else:
            raise Exception("Unknown initialization mode: '%s'" % self.initialization_mode)

        if self.initialization_mode == 'HierarchicalClustering':
            self.seed = long(environ['PMC_SEED'])
            self.analysis = environ['PMC_ANALYSIS']
            self.chunk_size = int(environ['PMC_CHUNKSIZE'])
            self.final_chunk_size = int(environ['PMC_FINAL_CHUNKSIZE'])
            self.max_steps = int(environ['PMC_MAX_STEPS'])
            self.dof = float(environ['PMC_DOF'])

        ###
        # book keeping
        ###
        self.job_infos = {}
        self.clean_files = []

        # to make jobscript run
        self.prefix = '#! /bin/bash\n\n'

        # never quite know if exception leads to __del__ being called
        # but if the final step of merges 1500 files don't want to lose
        # those precious files.
        # So clean only if things went well, and clean() called explicitly
#    def __del__(self):
#        self.clean()

    def clean(self):
        """
        Remove temporary files like job scripts, HDF5 files with intermediate results...
        """

        if not self.clean_up:
            return

        if self.verbose:
            print("Trying to remove:")
            print(self.clean_files)

        for file_name in self.clean_files:
            if os.path.exists(file_name):
                os.remove(file_name)

        self.clean_files = []

        if self.verbose:
            print("Trying to remove all job scripts:")
            print(self.job_infos.values())

        for j in self.job_infos.itervalues():
            if os.path.exists(j.script_name):
                os.remove(j.script_name)
            if os.path.exists(j.file_name):
                os.remove(j.file_name)

    def create_samples_file_name(self):
        return self.output_base + "pmc_parameter_samples_%d.hdf5" % self.step

    def create_job_script(self, sample_input_file, job_info, job_index):
        """
        Create a bash script, ready for queue submission,
        that computes the posterior/observables for a subrange
        of available samples
        """

        cmd = ""

        if self.initialization_mode == 'UncertaintyPropagation':
            cmd += 'eos-propagate-uncertainty'
        else:
            cmd += 'eos-scan-mc'
        cmd += ' ' + self.general_options
        cmd += ' --parallel %d' % self.thread_parallelization
        cmd += ' --output ' +  job_info.file_name
        if self.initialization_mode == 'UncertaintyPropagation':
            cmd += ' --pmc-input %s %d %d' % (self.uncertainty_input, job_info.min, job_info.max)
            cmd += ' --pmc-sample-directory %s' % self.uncertainty_sample_directory
            cmd += self.uncertainty_analysis
        else:
            cmd += ' --seed %d' % (self.seed + 1000 * self.step  + 100 * job_index)
            cmd += ' --chunks 1'
            cmd += ' --chunk-size %d' % self.chunk_size
            cmd += ' --use-pmc '
            cmd += ' ' + str(self.convergence_options)
            cmd += ' --pmc-calculate-posterior %s %d %d' % (sample_input_file, job_info.min, job_info.max)
            cmd += ' --pmc-final-chunksize 0'
            cmd += ' --pmc-dof %g' % self.dof
            cmd += ' ' + self.analysis

        # put every word in double quotes to be sure.
        # Only works if pathnames do not contain whitespace
        cmd = cmd.split()
        cmd = self.prefix + r'"' + r'" "'.join(cmd) + r'"'

        script_name = self.output_job_base + "pmc_job_%d_%d.sh" % (self.step, job_index)
        job_info.script_name = script_name

        f = open(script_name, 'w')
        f.write(cmd + '\n')
        f.close()

        # make the script executable
        os.chmod(script_name, 0755)

    def create_update_script(self, input_file, output_file, update=True, draw_samples=True, n_crop=None):
        """
        Create update script, return the script's file name.

        Keyword arguments:
        input_file -- contains the proposal density, samples and their weights
        output_file -- will contain new proposal density after update is performed
        draw_samples -- draw samples from new proposal (default: True)
        n_crop -- crop N samples with highest weight, overrides general options (default: None)

        """
        cmd = 'eos-scan-mc'
        cmd += ' --seed %d' % (self.seed + 1000 * self.step)
        cmd += ' ' + self.general_options
        cmd += ' --chunks 1'
        cmd += ' --chunk-size %d' % self.chunk_size
        cmd += ' --parallel %d' % self.thread_parallelization
        cmd += ' --use-pmc'
        cmd += ' ' + str(self.convergence_options)
        if update:
            cmd += ' --pmc-update %s' % input_file
        if draw_samples:
            cmd += ' --pmc-draw-samples'
            cmd += ' --pmc-initialize-from-file %s' % input_file
        if n_crop is not None:
            # in final step, want to know overall fit to all samples, so would turn off cropping
            cmd += ' --pmc-crop-highest-weights %d' % n_crop
        cmd += ' --pmc-final-chunksize %d' % self.final_chunk_size
        cmd += ' --pmc-dof %g' % self.dof
        cmd += ' --pmc-adjust-sample-size %d' % self.adjust_sample_size
        cmd += ' --output ' + output_file
        cmd += ' ' + self.analysis

        #put every word in quotes to be sure. Only works if arguments do not contain whitespace
        cmd = cmd.split()
        cmd = self.prefix + r'"' + r'" "'.join(cmd) + r'"'

        script = self.output_job_base + "pmc_update_%d.sh" % self.step
        self.clean_files.append(script)

        f = open(script, 'w')
        f.write(cmd + '\n')
        f.close()

        # make the script executable
        os.chmod(script, 0755)

        return script

    def extract_sample_info(self, file_name):
        """
        Extract the number of parameter samples
        from the HDF5 file.
        """

        import h5py
        import time

        n_steps = 6
        interval = 10

        for i in range(n_steps):
            try:
                f = h5py.File(file_name, 'r')
                if self.initialization_mode == 'UncertaintyPropagation':
                    dir = self.uncertainty_sample_directory
                else:
                    dir = '/data'
                n_samples = len(f[dir + '/samples'])
                f.close()
                return n_samples
            except:
                time.sleep(interval)

        raise Exception('Could not open %s, tried %d times and waited %d s in between each trial' % (file_name, n_steps, interval))

    def extract_statistics_info(self, file_name):
        """
        Find out if run converged.
        """

        import h5py
        import time

        converged = False

        n_steps = 6
        interval = 10

        for i in range(n_steps):
            try:
                f = h5py.File(file_name, 'r')
                converged = int(f['/data/statistics'].attrs['converged'])
                print('Statistics in step %d, converged: %d' % (self.step, converged))
                print('perplexity: %g, ESS: %g , evidence: %g' % tuple(f['/data/statistics'][-1]))
                f.close()
                return converged
            except:
                time.sleep(interval)

        raise Exception('Could not open %s, tried %d times and waited %d s in between each trial' % (file_name, n_steps, interval))

    def first_step(self, input_file=None, final=False):
        """
        Generate first set of samples, with PMC initialized from hierarchical clustering.
        """

        output_file = self.create_samples_file_name() #output_base + "_parameter_samples_%d.hdf5" % self.step

        cmd = 'eos-scan-mc '
        cmd += ' --seed %d' % self.seed
        cmd += ' ' + self.general_options
        cmd += ' ' + self.initialization_options
        cmd += ' ' + self.ignore_groups
        cmd += ' --pmc-group-by-r-value %g' % self.group_by_r_value
        cmd += ' --chunks 1'
        cmd += ' --chunk-size %d' % self.chunk_size
        cmd += ' --use-pmc '
        cmd += ' --pmc-dof %g' % self.dof
        if input_file:
            cmd += ' --pmc-initialize-from-file ' + input_file
            cmd += ' --pmc-final-chunksize %d' % self.final_chunk_size
            if final:
                cmd += ' --pmc-final 1'
        elif self.initialization_mode == 'HierarchicalClustering':
            cmd += ' --pmc-initialize-from-file ' + self.prerun_file_name
            cmd += ' --pmc-hierarchical-clusters %d' % self.target_number_of_clusters
            cmd += ' --global-local-covariance-window %d' % self.patch_length
            cmd += ' --global-local-skip-initial %g' % self.skip_initial
            cmd += ' --pmc-final-chunksize 0'
        else:
            raise Exception("Found neither PMC dump nor global local nor MCMC prerun [merge] input file")
        cmd += ' ' + str(self.convergence_options)
        cmd += ' --output ' + output_file
        cmd += ' --pmc-draw-samples'
        cmd += ' ' + self.analysis

        #put every word in quotes to be sure. Only works if arguments do not contain whitespace
        cmd = cmd.split()
        cmd = r'"' + r'" "'.join(cmd) + r'"'

        print("Running first step with initialization from %s" % self.initialization_mode if input_file is None else input_file)
        status, output = commands.getstatusoutput(cmd)

        log_file_name = self.output_job_base + "pmc_step_%d.log" % self.step
        f = open(log_file_name, 'w')
        f.write(cmd)
        f.write('\n\n')
        f.write(output)

        if status != 0:
            raise Exception("First step failed with status %d and output:\n %s" % (status, output))

        return output_file

    def init_jobs(self, n_samples, sample_input_file, output_scripts=True):

        job_infos = {}

        # construct intervals of same length
        # if n_samples not divisible, last job gets remaining samples
        samples_per_job = int(n_samples) / int(self.number_of_jobs)

        if samples_per_job < 1:
            raise Exception("samples per job at 0. Choose more samples or less jobs")

        print("samples_per_job: %d" % samples_per_job)

        for i in range(self.number_of_jobs):
            job_infos[i] = JobInfo(i * samples_per_job, (i + 1) * samples_per_job, self.output_base + "pmc_job_%d_%d.hdf5" % (self.step, i))

        job_infos[self.number_of_jobs - 1].max = n_samples

        if output_scripts:
            for i in job_infos.keys():
                self.create_job_script(sample_input_file, job_infos[i], i)

        self.job_infos = job_infos

    def merge(self, output_file_name):
        """
        Merge PMC results
        """

        import h5py
        import tables as pytables

        merge_file_name = output_file_name + '_merge'

        output_file = h5py.File(output_file_name, 'r')
        weight_record = None

        # keep track of which weights are to be ignored
        broken = np.zeros((output_file['/data/samples'].len(),),dtype=np.int8)

        # check length of each output file
        # and prepare to sort filename according to index
        min_index_job_index = {}
        for i,j in self.job_infos.iteritems():
            min_index_job_index[j.min] = i

            # if anything goes wrong with opening the file, we assume samples should be ignored
            try:
                f = h5py.File(j.file_name, 'r')
                assert(j.max - j.min == f['/data/weights'].len())
                weight_record = np.array(f['/data/weights'][0])
                f.close()
            except:
                print("Corrupt file: %s" % j.file_name)
                j.broken = True
                broken[j.min:j.max] = 1

        if weight_record is None:
            raise Exception("Could not find a single sane sample file")
        else:
            #default value for posterior and weight = -inf, so w = exp(log(0)) = 0
            weight_record['posterior'] = -np.inf #np.log(0.0)
            weight_record['weight'] = -np.inf    #np.log(0.0)
        n_broken = float(len(np.where(broken == 1)[0]))
        print("Percentage of broken samples: %g out of %d =  %g " % ( n_broken, len(broken), n_broken / len(broken)))

        sorted_keys= min_index_job_index.keys()
        sorted_keys.sort()

        # copy samples to the merge file
        merge_file = h5py.File(merge_file_name, 'w')
        merge_file.create_group('/data')
        merge_file.create_group('/data/initial')
        # copy samples to merge file
        output_file.copy('/data/samples', merge_file['/data'])
        output_file.copy('/descriptions/', merge_file['/'])
        try:
            output_file.copy('/data/statistics', merge_file['/data'])
            output_file.copy('/data/initial/components', merge_file['/data/initial'])
        except:
            # in first update, no previous stats available
            output_file.copy('/data/components', merge_file, name='/data/initial/components')

        merge_file.close()
        output_file.close()
        self.clean_files.append(output_file_name)

        # sort files such that the file computing sample 0 comes first...
        # not needed if jobs aren't broken up into subparts
        #            weight_file_names = [self.job_infos[min_index_job_index[i]].file_name for i in sorted_keys]
        #            broken_files = [self.job_infos[min_index_job_index[i]].broken for i in sorted_keys]
        #            merge_datasets(output=merge_file_name, groups=['/data/weights'], \
            #                           input_files=weight_file_names)

        # merge using pytables
        merge_file = pytables.openFile(merge_file_name, "r+")
        # merge using h5py
#         merge_file = h5py.File(merge_file_name, "w")
#         data_grp = merge_file.create_group('data')
#         weights_data = np.array([])

        # repeat the sample line if needed to make up enough records
        if self.job_infos[0].broken:
            weight_data_set = merge_file.createTable('data', 'weights', \
                                                     np.repeat(weight_record, self.job_infos[0].max - self.job_infos[0].min, axis=0))
            #weights_data = np.repeat(weight_record, self.job_infos[0].max - self.job_infos[0].min, axis=0)
        else:
            f = pytables.openFile(self.job_infos[0].file_name, 'r')
            weight_data_set = merge_file.createTable('/data', 'weights', f.getNode('/data/weights').read())
            #f = h5py.File(self.job_infos[0].file_name, 'r')
            #weights_data = f['/data/weights'].value
            f.close()

        for i in self.job_infos.keys()[1:]:
            j = self.job_infos[i]
            if j.broken:
                print("Filling in broken record %d times" % (j.max - j.min))
                weight_data_set.append(np.repeat(weight_record, j.max - j.min, axis=0))
                #weights_data.append(np.repeat(weight_record, j.max - j.min, axis=0))
            else:
                f = pytables.openFile(j.file_name, 'r')
                weight_data_set.append(f.getNode('/data/weights').read())
                #f = h5py.File(j.file_name, 'r')
                #weights_data.append(f['/data/weights'].value)
                f.close()

        #data_grp.create_dataset('weights', data = weight_data)
        merge_file.close()

        # add information on broken files
        merge_file = h5py.File(merge_file_name, 'r+')
        merge_file['/data'].create_dataset("broken", data=broken)
        merge_file.close()

        return merge_file_name

    def merge_uncertainty(self):
        """
         merge results:
         o copy weights from input
         o copy observables' values into a single data set.
         o If file is broken set weight and observables to zero
        """
        import h5py
        import tables as pytables

        merge_file_name = self.output_base + 'unc.hdf5'

        # determine file type: monolithic vs queue output
        input_file = h5py.File(self.uncertainty_input, 'r')

        n_samples = input_file[self.uncertainty_sample_directory + '/samples'].len()

        # keep track of which weights are to be ignored

        broken = np.zeros((n_samples,),dtype=np.int8)

        # keep track of first sane file
        found_sane = None
        for i,j in self.job_infos.iteritems():

            # if anything goes wrong with opening the file, we assume samples should be ignored
            try:
                f = h5py.File(j.file_name, 'r')
                assert(j.max - j.min == f['/data/observables'].len())
                if found_sane is None:
                    found_sane = i
                f.close()
            except:
                print("Corrupt file: %s" % j.file_name)
                j.broken = True
                broken[j.min:j.max] = 1

        if found_sane is None:
            raise Exception("Could not find a single sane sample file")

        n_broken = float(len(np.where(broken == 1)[0]))
        print("Percentage of broken samples: %g out of %d =  %g " % ( n_broken, len(broken), n_broken / len(broken)))

        # create data arrays in memory first
        if self.uncertainty_sample_directory == '/data/final':
            weights_array = np.zeros(n_samples, dtype=np.dtype([('posterior', float), ('weight', float)]))
            weights_array.T['posterior'] = input_file['/data/final/samples'][:].T[-2]
            weights_array.T['weight'] = input_file['/data/final/samples'][:].T[-1]
        else:
            weights_array = np.array(input_file['/data/weights'][:])
        assert(len(weights_array) == n_samples)

        f = h5py.File(self.job_infos[found_sane].file_name, 'r')
        # Repeat first line to have n_samples rows, and the same #columns as observables
        observables_array = np.tile(f['/data/observables'][0], (n_samples,1))
        f.close()

        for j in self.job_infos.itervalues():
            print(j.file_name)
            if j.broken:
                print("Filling in broken record %d times" % (j.max - j.min))
                # set zero weight, but on log scale
                weights_array[j.min:j.max].T['posterior'] = -np.inf #np.log(0.0)
                weights_array[j.min:j.max].T['weight'] = -np.inf    #np.log(0.0)
                # account for the case of a single observable
                n_cols = 1
                if len(observables_array.shape) > 1:
                    n_cols = observables_array.shape[1]
                for col in range(n_cols):
                    observables_array[j.min:j.max].T[col] = 0.0
            else:
                f = h5py.File(j.file_name, 'r')
                observables_array[j.min:j.max] = f['/data/observables'][:]
                f.close()

        # dump data to disk
        merge_file = h5py.File(merge_file_name, 'w')

        merge_file.create_group('data')
        merge_file['/data'].create_dataset("observables", data=observables_array, compression='gzip')
        merge_file['/data'].create_dataset("weights", data=weights_array, compression='gzip')

        merge_file.create_group('descriptions')
        input_file.copy('/descriptions/constraints', merge_file['/descriptions'])
        input_file.copy('/descriptions/parameters', merge_file['/descriptions'])
        f = h5py.File(self.job_infos[found_sane].file_name, 'r')
        f.copy('/descriptions/observables', merge_file['/descriptions'])

        f.close()
        merge_file.close()
        input_file.close()

    def run(self, first_step=None, update=None, samples=None, final=False,
            step=None, input_file=None, force_final=False, merge=False):
        """
        Start the loop and run PMC
        """
        if step is not None:
            self.step = step
        converged = False

        if merge and input_file:
            n_samples = self.extract_sample_info(input_file)
            self.init_jobs(n_samples, input_file, output_scripts=False)
            merge_file_name = self.merge(input_file)
            return

        if samples:
            output_file = self.create_samples_file_name()
        # default: run first step with clustering
        elif not update or input_file:
            output_file = self.first_step(input_file, final)
            if first_step:
                return

        while self.step < self.max_steps and not final:

            # start jobs
            if update:
                output_file, converged = self.update(self.create_samples_file_name() + '_merge')
                update = None
            else:
                output_file, converged = self.single_step(output_file)

            self.clean()

            # break on convergence or max. number of steps
            if converged:
                print("Convergence achieved after %d update steps" % self.step)
                final = True

        if not final and not converged:
            sys.stderr.write("Performed %d steps, but PMC did NOT converge\n" % self.max_steps)

        if final or force_final:
            output_file, converged = self.single_step(output_file, final=True)
        self.clean()

    def run_single_jobs(self):
        raise Exception("PMC_Iterator.run_single_jobs not implemented yet")

    def run_uncertainty(self, n_samples=None):
        """
        Massively parallelize uncertainty propagation based on PMC output
        """
        assert(self.initialization_mode == 'UncertaintyPropagation')
        if n_samples is not None:
            n_samples = int(n_samples)
        else:
            n_samples = self.extract_sample_info(self.uncertainty_input)
        self.init_jobs(n_samples, self.uncertainty_input)
        self.run_single_jobs()
        self.merge_uncertainty()
        self.clean()

    def run_update_job(self, script_name):
        """
        Run the call to update the PMC proposal function and optionally to draw samples
        in the queue.
        Raise an exception if job did not succeed.
        """
        raise NotImplementedError("PMC_Iterator.run_update_job() not implemented yet")

    def single_step(self, input_file, final=False):
        """ Perform one PMC step"""
        n_samples = self.final_chunk_size if final else self.extract_sample_info(input_file)
        self.init_jobs(n_samples, input_file)
        self.run_single_jobs(final=final)
        merge_file_name = self.merge(input_file)
        return self.update(merge_file_name, draw_samples=not final)

    def update(self, merge_file_name, draw_samples=True, crop=None):
        """
        update PMC proposal function and draw new samples for next step.
        """
        self.step += 1

        new_samples_output_file_name = self.create_samples_file_name()
        script_name = self.create_update_script(merge_file_name, new_samples_output_file_name,
                                                draw_samples=draw_samples, n_crop=crop)

        print('Updating PMC proposal ' + 'and drawing samples' if draw_samples else '')

        # use standard queue for final update with possibly many samples
        final = not draw_samples
        #        self.run_update_job(script_name, final)
        if final:
            print("Final update temporarily disabled due to seg fault (unconfirmed for 1D with 1e6 samples)")
            print("But final update not needed anymore, compute perplexity and ESS w/o pmclib")
        else:
            self.run_update_job(script_name, final)

        # check if converged
        if final:
            # don't wait for final update, may take very long
            converged = True
        else:
            converged = self.extract_statistics_info(new_samples_output_file_name)

        return new_samples_output_file_name, converged

class CondorChecker(TaskThread):

    def __init__(self, cluster_id):
        super(CondorChecker, self).__init__()

        self.cluster_id = cluster_id

    def task(self):
        """
        Check if a cluster has finished yet.
        """

        # poll status
        status, output = commands.getstatusoutput('condor_q %s' % self.cluster_id)

        print(output)

        job_running = False
        for line in output.splitlines():
            if 'beaujean' in line:
                job_running = True

        # all jobs finished successfully
        if not job_running:
            print('All jobs seem to be finished')
            self.shutdown()

class CondorOptions(object):
    """
    Collect all options particular to the Condor queue
    """

    def __init__(self):
        self.number_of_dag_retries = 5
        self.polling_time = 5 # in seconds

class CondorIterator(PMC_Iterator):

    def __init__(self):
        super(CondorIterator, self).__init__()

        # somehow need job scripts in home directory?
        self.output_job_base = environ['HOME'] + '/job_scripts/'
        self.options = CondorOptions()

    def control_jobs(self, cluster_id):
        """
        Handle job execution in queue
        """

        print('Polling job with id %s every %d second'  % (cluster_id, self.condor_options.polling_time))

        checker = CondorChecker(cluster_id)
        checker.setInterval(self.condor_options.polling_time)
        checker.run()

        print('Finished controlling')

    def run_single_jobs(self):
        """
        Start jobs in the batch queue.
        """

        ###
        # create DAG file
        ###

        cmb_file_name = self.output_job_base + '.cmb'
        cmb_merge_file_name = self.output_job_base + '_merge.cmb'
        condor_log_file_name = self.output_job_base + '_condor.log'

        dag = ''
        dag += 'JOB merge %s \n' % cmb_merge_file_name
        for i,j in self.job_infos.iteritems():
            dag += 'JOB J%d %s \n' % (i, cmb_file_name)
            dag += 'PARENT J%d CHILD merge \n' % i
            dag += 'RETRY J%d %d \n' % (i, self.condor_options.number_of_dag_retries)
            dag += 'VARS J%d script_name=\"%s\" \n' % (i, j.script_name)

        dag_file_name = self.output_job_base + '.dag'
        f = open(dag_file_name, 'w')
        f.write(dag)
        f.close()

        ###
        # create ordinary condor submit file
        ###

        cmd = ''
        cmd += 'Error        = ' + self.output_job_base + '_$(cluster).log \n'
        cmd += 'Executable = $(script_name) \n'
        cmd += 'Getenv       = True \n'
        cmd += 'Log          = %s \n' % condor_log_file_name
        cmd += 'Notification = NEVER \n'
        cmd += 'Output       = ' + self.output_job_base + '_$(cluster).log \n'
        cmd += 'Requirements = (Pool=="Theory") \n' #todo SlotId
        cmd += 'Universe     = vanilla \n'
        cmd += 'Queue \n'

        f = open(cmb_file_name, 'w')
        f.write(cmd)
        f.close()

        ###
        # create merge condor job script
        ###

        cmd = ''
        cmd += 'Error        = ' + self.output_job_base + '_merge.log \n'
        cmd += 'Executable   =  ' + environ['HOME'] + '/workspace/Sandbox/eos/jobs/job_manager_merge.sh \n' #todo remove hard coded name
        cmd += 'Getenv       = True \n'
        cmd += 'Log          = %s \n' % condor_log_file_name
        cmd += 'Notification = NEVER \n'
        cmd += 'Output       = ' + self.output_job_base + '_merge.log \n'
        cmd += 'Requirements = (Pool=="Theory" && SlotID==2) \n'
        cmd += 'Universe     = vanilla \n'
        cmd += 'Queue \n'

        f = open(cmb_merge_file_name, 'w')
        f.write(cmd)
        f.close()

        ###
        # submit jobs to Condor queue
        ###

        status, output = commands.getstatusoutput('condor_submit_dag -f %s' % dag_file_name)
        if status != 0:
            raise Exception("Couldn't submit condor dag job: \n %s" % output)

        # extract cluster number
        cluster_id = None
        for line in output.splitlines():
            if 'submitted to cluster' in line:
                cluster_id = line.split()[-1][:-1]

        if cluster_id is None:
            raise Exception("Could extract cluster id from:\n %s" % output)

        self.control_jobs(cluster_id)

###
### Classes for "SGE"-based clusters
###
class SGE_Checker(TaskThread):

    def __init__(self, job_ids, interval=15, check_error_status=False):
        super(SGE_Checker, self).__init__()
        self.job_ids = job_ids
        self.setInterval(interval) # in seconds
        self.check_error_status = check_error_status

    def task(self):
        """
        Check if all jobs finished yet. Sample output to parse is
        job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID
        -----------------------------------------------------------------------------------------------------------------
        2750697 0.55010 Scenario2_ fdb          r     01/25/2012 20:09:49 standard@ot10.t2.rzg.mpg.de        1
        """

        # poll status
        status, output = commands.getstatusoutput('qstat -u fdb')
        #        print(output)

        # extract job ids from first column, ignoring the first two rows
        running_job_ids = [int(line.split()[0]) for line in output.splitlines()[2:]]
        job_status = [line.split()[4] for line in output.splitlines()[2:]]

        job_running = False
        for job_number, id in self.job_ids.iteritems():
            try:
                # is job id still in queue?
                i = running_job_ids.index(id)
                status = job_status[i]
                if status == 'r' or status == 'qw':
                    job_running = True
                    sys.stdout.write('Job %d still active with status %s. ' % (id, status))
                    break
                if status == 'Eqw':
                    sys.stderr.write('Job %d is in error state Eqw. ' % id)
                    self.shutdown(job_number)
                    return
            except ValueError:
                pass

        # all jobs finished successfully
        if job_running:
            return

        print('All jobs seem to be finished;')
        print("Checking error? %s" % str(self.check_error_status))

        return_value = 0

        if not self.check_error_status:
            self.shutdown(return_value=return_value)
            return

        # check if there was an error code somewhere
        for job_number, id in self.job_ids.iteritems():
            while True:
                status, output = commands.getstatusoutput('qacct -j %d' % id)
                print("Checking status of job %d" % id)
                if status != 0:
                    sys.stderr.write('Could not retrieve return code of job %d. Waiting 5 seconds\n' % id)
                    time.sleep(5)
                    continue

                for line in output.splitlines():
                    if 'exit_status' in line:
                        job_ret_code = int(line.split()[1])

                        if job_ret_code != 0:
                            return_value = job_ret_code
                            sys.stderr.write("Job %d finished with non-zero return code %d\n" % (id, job_ret_code))

                # successfully checked this job
                break

        self.shutdown(return_value=return_value)

class SGE_Options(object):
    """
    Collect all options particular to the SGE queue
    """

    def __init__(self):
        self.queue = env('SGE_QUEUE', 'standard')
        self.final_queue = env('SGE_FINAL_QUEUE', 'standard')
        self.check_error_status = bool(env('SGE_CHECK_ERROR_STATUS', False))

class SGE_Iterator(PMC_Iterator):

    def __init__(self):
        super(SGE_Iterator, self).__init__()

        # somehow need job scripts in home directory?

        if not os.path.isdir(self.output_job_base):
            os.makedirs(self.output_job_base)

        self.output_job_base = os.path.join(self.output_job_base, os.path.split(self.output_base)[1])

        self.options = SGE_Options()

    def control_jobs(self, job_ids):
        """
        Handle job execution in queue in abstract fashion.
        Returns once all jobs are finished.
        """

        print('Polling SGE every %d seconds' % self.polling_interval)

        checker = SGE_Checker(job_ids, self.polling_interval)
        ret_value = checker.run()

        if ret_value:
            sys.stderr.write("Some job did NOT finish alright, but had return code %d\n" % ret_value)
#            self.start_job(ret_value)
        else:
            print('Finished controlling. All jobs returned successfully.')

    def run_single_jobs(self, final=False):
        """
        Start jobs in the batch queue.
        """

        job_ids = {}
        # submit and store job id
        for i in self.job_infos.iterkeys():
            job_ids[i] = self.start_job(i, final=final)
        print("Running the following jobs:")
        print(job_ids)

        self.control_jobs(job_ids)

    def run_update_job(self, script_name, final=False):
        log_file_name = self.output_job_base + 'pmc_update_%d' % self.step + '.log'
#        self.clean_files.append(log_file_name)

        cmd =  'qsub'
        cmd += ' -q %s' % (self.options.final_queue if final else self.options.queue)
        cmd += ' -j y' # join stderr with stdout
        cmd += ' -N job_update_%d' % self.step #job_indentifier
        cmd += ' -o %s' % log_file_name
        cmd += ' -V'
        cmd += ' -S /bin/bash'
        cmd += ' %s' % script_name

        status, output = commands.getstatusoutput(cmd)

        if status != 0 or not output:
            raise Exception("Couldn't submit to SGE queue: \n Tried '%s'\n\n and received:\n\n %s" % (cmd, output))

        # parse job id from a line like
        # 'Your job 2756454 ("empty_job.sh") has been submitted'
        start = output.find('job')
        end = output.find('("')
        job_id = int(output[start + 4: end])

        # don't wait for job to finish
        if final:
            print("Final update job %d is on its way, I don't wait until it is finished" % job_id)
            return

        self.control_jobs({0:job_id})

    def start_job(self, job_index, final=False):
        """
        Start a single job with the given index.
        Returns the job id assigned by SGE
        """

        log_file_name = self.output_job_base + '_%d' % job_index + '.log'
        self.clean_files.append(log_file_name)

        cmd =  'qsub'
        cmd += ' -q %s' % (self.options.final_queue if final else self.options.queue)
        cmd += ' -j y' # what does it mean?
        cmd += ' -N job_%d_%d' % (self.step, job_index) #job_indentifier
        cmd += ' -e %s' % log_file_name
        cmd += ' -o %s' % log_file_name
        cmd += ' -V'
        cmd += ' -S /bin/bash'
        cmd += ' %s' % self.job_infos[job_index].script_name

        status, output = commands.getstatusoutput(cmd)

        if status != 0 or not output:
            raise Exception("Couldn't submit to SGE queue: \n Tried '%s'\n\n and received:\n\n %s" % (cmd, output))

        # remove file immediately, sometimes get this error when removing due to incredibly slow file system:
        """OSError: [Errno 110] Connection timed out:"""
        try:
            if os.path.exists(self.job_infos[job_index].script_name):
                pass
            #                os.remove(self.job_infos[job_index].script_name)
        except OSError:
            pass

        # parse job id from a line like
        # 'Your job 2756454 ("empty_job.sh") has been submitted'
        start = output.find('job')
        end = output.find('("')
        return int(output[start + 4: end])

def restart_jobs():
    """Submit jobs after an OSError has stopped program"""
    output_job_base = '/afs/ipp/home/f/fdb/JobOutput/2012-06-25/Scenario1_all_nuis/sc1_all_nuis'
    for i in range(481, 600):
        log_file_name = output_job_base + ('_%d' % i) + '.log'

        cmd =  'qsub'
        cmd += ' -q %s' % "short"
        cmd += ' -j y'
        cmd += ' -N job_%d_%d' % (1, i) #job_indentifier
        cmd += ' -e %s' % log_file_name
        cmd += ' -o %s' % log_file_name
        cmd += ' -V'
        cmd += ' -S /bin/bash'
        cmd += ' %s' % output_job_base + ("_job_1_%d.sh" % i)

        if i == 481:
            print(cmd)

        status, output = commands.getstatusoutput(cmd)

        if status != 0 or not output:
            raise Exception("Couldn't submit to SGE queue: \n Tried '%s'\n\n and received:\n\n %s" % (cmd, output))

###
### Classes for "Slurm"-based clusters
###
class Slurm_Checker(TaskThread):

    def __init__(self, job_ids, interval=15, check_error_status=False):
        super(Slurm_Checker, self).__init__()
        self.job_ids = job_ids
        self.setInterval(interval) # in seconds
        self.check_error_status = check_error_status

    def task(self):
        """
        Check if all jobs finished yet. Sample output to parse is
        JOBID PARTITION     NAME     USER   ST       TIME   NODES NODELIST(REASON)
        XXXXX default       xx       vandyk R        aa::bb 1     nodexxx
        """

        # poll status
        status, output = commands.getstatusoutput('squeue -u vandyk')
        #        print(output)

        # extract job ids from first column, ignoring the first two rows
        running_job_ids = [int(line.split()[0]) for line in output.splitlines()[1:]]
        job_status = [line.split()[4] for line in output.splitlines()[1:]]

        job_running = False
        for job_number, id in self.job_ids.iteritems():
            try:
                # is job id still in queue?
                i = running_job_ids.index(id)
                status = job_status[i]
                if status == 'R' or status == 'CG' or status == 'CF' or status == 'PD':
                    job_running = True
                    sys.stdout.write('Job %d still active with status %s. ' % (id, status))
                    break
                if status == 'F':
                    sys.stderr.write('Job %d has failed' % id)
                    self.shutdown(job_number)
                    return
                if status == 'CA':
                    sys.stderr.write('Job %d has been cancelled' % id)
                    self.shutdown(job_number)
                    return
                if status == 'TO':
                    sys.stderr.write('Job %d has timed out' % id)
                    self.shutdown(job_number)
                    return
            except ValueError:
                pass

        # all jobs finished successfully
        if job_running:
            return

        print('All jobs seem to be finished;')
        print("Checking error? %s" % str(self.check_error_status))

        return_value = 0

        if not self.check_error_status:
            self.shutdown(return_value=return_value)
            return

        # check if there was an error code somewhere
        for job_number, id in self.job_ids.iteritems():
            while True:
                status, output = commands.getstatusoutput('sacct -j %d --format ' % id)
                print("Checking status of job %d" % id)
                if status != 0:
                    sys.stderr.write('Could not retrieve return code of job %d. Waiting 5 seconds\n' % id)
                    time.sleep(5)
                    continue

                for line in output.splitlines():
                    lid, lec = line.split()
                    if str(id) == lid:
                        sec, ssig = lec.split(':')
                        job_ret_code = int(sec)

                        if job_ret_code != 0:
                            return_value = job_ret_code
                            sys.stderr.write("Job %d finished with non-zero return code %d\n" % (id, job_ret_code))

                # successfully checked this job
                break

        self.shutdown(return_value=return_value)

class Slurm_Options(object):
    """
    Collect all options particular to the Slurm queue
    """

    def __init__(self):

        try:
            self.queue = environ['SLURM_QUEUE']
        except KeyError:
            self.queue = 'medium'

        try:
            self.final_queue = environ['SLURM_FINAL_QUEUE']
        except KeyError:
            self.final_queue = 'defq'

        try:
            self.check_error_status = bool(environ['SLURM_CHECK_ERROR_STATUS'])
        except KeyError:
            self.check_error_status = False

class Slurm_Iterator(PMC_Iterator):

    def __init__(self):
        super(Slurm_Iterator, self).__init__()

        # somehow need job scripts in home directory?

        if not os.path.isdir(self.output_job_base):
            os.makedirs(self.output_job_base)

        self.output_job_base = os.path.join(self.output_job_base, os.path.split(self.output_base)[1])

        self.options = Slurm_Options()

    def control_jobs(self, job_ids):
        """
        Handle job execution in queue in abstract fashion.
        Returns once all jobs are finished.
        """

        print('Polling Slurm every %d seconds' % self.polling_interval)

        checker = Slurm_Checker(job_ids, self.polling_interval)
        ret_value = checker.run()

        if ret_value:
            sys.stderr.write("Some job did NOT finish alright, but had return code %d\n" % ret_value)
#            self.start_job(ret_value)
        else:
            print('Finished controlling. All jobs returned successfully.')

    def run_single_jobs(self, final=False):
        """
        Start jobs in the batch queue.
        """

        job_ids = {}
        # submit and store job id
        for i in self.job_infos.iterkeys():
            job_ids[i] = self.start_job(i, final=final)
        print("Running the following jobs:")
        print(job_ids)

        self.control_jobs(job_ids)

    def run_update_job(self, script_name, final=False):
        log_file_name = self.output_job_base + '_update_%d' % self.step + '.log'
#        self.clean_files.append(log_file_name)

        cmd =  'sbatch'
        cmd += ' --partition=%s' % (self.options.final_queue if final else self.options.queue)
        cmd += ' --time=%s' % ('10:00:00' if final else '4:00:00')
        cmd += ' -J job_update_%d' % self.step #job_indentifier
        cmd += ' -o %s' % log_file_name
        cmd += ' -e %s.err' % log_file_name
        cmd += ' -v'
        cmd += ' %s' % script_name

        status, output = commands.getstatusoutput(cmd)

        if status != 0 or not output:
            raise Exception("Couldn't submit to Slurm queue: \n Tried '%s'\n\n and received:\n\n %s" % (cmd, output))

        # parse job id from a line like
        # 'Submitted batch job 451849'
        job_id = int(output.split()[-1])

        # don't wait for job to finish
        if final:
            print("Final update job %d is on its way, I don't wait until it is finished" % job_id)
            return

        self.control_jobs({0:job_id})

    def start_job(self, job_index, final=False):
        """
        Start a single job with the given index.
        Returns the job id assigned by Slurm
        """

        log_file_name = self.output_job_base + '_%d' % job_index + '.log'
        self.clean_files.append(log_file_name)

        cmd =  'sbatch'
        #cmd += ' -q %s' % (self.options.final_queue if final else self.options.queue)
        cmd += ' -J job_%d_%d' % (self.step, job_index) #job_indentifier
        cmd += ' -e %s' % log_file_name
        cmd += ' -o %s' % log_file_name
        cmd += ' -v'
        cmd += ' %s' % self.job_infos[job_index].script_name

        status, output = commands.getstatusoutput(cmd)

        if status != 0 or not output:
            raise Exception("Couldn't submit to SGE queue: \n Tried '%s'\n\n and received:\n\n %s" % (cmd, output))

        # remove file immediately, sometimes get this error when removing due to incredibly slow file system:
        """OSError: [Errno 110] Connection timed out:"""
        try:
            if os.path.exists(self.job_infos[job_index].script_name):
                pass
            #                os.remove(self.job_infos[job_index].script_name)
        except OSError:
            pass

        # parse job id from a line like
        # 'Submitted batch job 451849'
        return int(output.split()[-1])

def restart_jobs():
    """Submit jobs after an OSError has stopped program"""
    output_job_base = '/scratch/vandyk/bayes2/JobOutput/2012-06-25/Scenario1_all_nuis/sc1_all_nuis'
    for i in range(481, 600):
        log_file_name = output_job_base + ('_%d' % i) + '.log'

        cmd =  'sbatch'
        #cmd += ' -q %s' % "short"
        cmd += ' -J job_%d_%d' % (1, i) #job_indentifier
        cmd += ' -e %s' % log_file_name
        cmd += ' -o %s' % log_file_name
        cmd += ' -v'
        cmd += ' %s' % output_job_base + ("_job_1_%d.sh" % i)

        if i == 481:
            print(cmd)

        status, output = commands.getstatusoutput(cmd)

        if status != 0 or not output:
            raise Exception("Couldn't submit to Slurm queue: \n Tried '%s'\n\n and received:\n\n %s" % (cmd, output))

###
### Classes for "loadleveler"-based clusters
###
class LL_Checker(TaskThread):

    def __init__(self, job_ids, interval=15, check_error_status=False):
        super(LL_Checker, self).__init__()
        self.job_ids = job_ids
        self.setInterval(interval) # in seconds
        self.check_error_status = check_error_status

    def task(self):
        """
        Check if all jobs finished yet. Sample output to parse is
Id                       Owner      Submitted   ST PRI Class        Running On
------------------------ ---------- ----------- -- --- ------------ -----------
xcat.163067.0            ru72xaf2    8/29 13:53 R  50  serial       n040
xcat.163081.0            ru72xaf2    8/29 13:53 I  50  serial
        """

        # poll status
        status, output = commands.getstatusoutput('llq -u ru72xaf2')

        # relevant lines ignore first two and last row
        rel_lines = output.splitlines()[2:-2]

        # split line into words
        running_job_ids = [int(line.split()[0].split('.')[1]) for line in rel_lines]
        job_status = [line.split()[4] for line in rel_lines]

        # check all registered jobs for their status
        job_running = False
        for job_number, id in self.job_ids.iteritems():
            try:
                # is job id still in queue?
                i = running_job_ids.index(id)
                status = job_status[i]
                if status == 'R' or status == 'I':
                    job_running = True
                    sys.stdout.write('Job %d still active with status %s. ' % (id, status))
                    break
            except ValueError:
                pass

        # all jobs finished successfully
        if job_running:
            return

        print('All jobs seem to be finished;')
        print("Checking error? %s" % str(self.check_error_status))

        return_value = 0

        if self.check_error_status:
            print("error status checking not implemented")

        self.shutdown(return_value=return_value)

class LL_Options(object):
    """
    Collect all options particular to the LL queue
    """

    def __init__(self):
        self.queue = env('LL_QUEUE', 'serial')
        self.final_queue = env('LL_FINAL_QUEUE', 'serial')
        self.check_error_status = False

class LL_Iterator(PMC_Iterator):

    def __init__(self):
        super(LL_Iterator, self).__init__()

        # somehow need job scripts in home directory?

        if not os.path.isdir(self.output_job_base):
            os.makedirs(self.output_job_base)

        self.output_job_base = os.path.join(self.output_job_base, os.path.split(self.output_base)[1])

        self.options = LL_Options()

    def submit(self, script_name, log_file_name, queue, job_name):
        """
        Create a job submission file and submit. Return job id.

        script_name: the script that is run by loadleveler
        queue: the name of the queue to which the job gets submitted

        """
        submission = """#! /bin/bash

#@ group =  pr85tu
#@ job_type = serial
#@ resources = ConsumableCpus(1)
#@ class = %s
###                    hh:mm:ss
#@ wall_clock_limit = 20:15:50
#@ node_usage=shared
#@ job_name = eos-%s
#@ initialdir = $(home)/workspace/eos-scripts/bayes2
#@ output = %s
#@ error  = %s
#@ notification=error
#@ notify_user=Frederik.Beaujean@lmu.de
#@ queue

%s
""" % (queue, job_name, log_file_name, log_file_name, script_name)
        ###
        # create file
        ###

        submit_file_name = self.output_job_base + "eos_ll.cmd"
        f = open(submit_file_name, 'w')
        f.write(submission)
        f.close()

        # make the script executable
        os.chmod(submit_file_name, 0755)

        ###
        # submit file
        ###
        cmd = 'llsubmit ' + submit_file_name
        status, output = commands.getstatusoutput(cmd)

        self.clean_files.append(submit_file_name)

        if status != 0 or not output:
            raise Exception("Couldn't submit to LL queue: \n Tried '%s'\n\n and received:\n\n %s" % (cmd, output))

        # extract job index from a line like
        # llsubmit: The job "xcat.163140" has been submitted.
        start = output.find('xcat.')
        end = output.find('"', start)
        job_id = int(output[start + 5: end])

        return job_id

    def control_jobs(self, job_ids):
        """
        Handle job execution in queue in abstract fashion.
        Returns once all jobs are finished.
        """

        print('Polling LL every %d seconds' % self.polling_interval)

        checker = LL_Checker(job_ids, self.polling_interval)
        ret_value = checker.run()

        if ret_value:
            sys.stderr.write("Some job did NOT finish alright, but had return code %d\n" % ret_value)
#            self.start_job(ret_value)
        else:
            print('Finished controlling. All jobs returned successfully.')

    def run_single_jobs(self, final=False):
        """
        Start jobs in the batch queue.
        """

        job_ids = {}
        # submit and store job id
        for i in self.job_infos.iterkeys():
            job_ids[i] = self.start_job(i, final=final)
        print("Running the following jobs:")
        print(job_ids)

        self.control_jobs(job_ids)

    def run_update_job(self, script_name, final=False):
        log_file_name = self.output_job_base + 'pmc_update_%d' % self.step + '.log'

        job_id = self.submit(script_name=script_name,
                         log_file_name=log_file_name,
                         queue=self.options.final_queue if final else self.options.queue,
                         job_name='job_%d' % self.step)

        # don't wait for job to finish
        if final:
            print("Final update job %d is on its way, I don't wait until it is finished" % job_id)
            return

        self.control_jobs({0:job_id})

#         cmd = """#! /bin/bash

# #@ group =  pr85tu
# #@ job_type = serial
# #@ class = %s
# ###                    hh:mm:ss
# ##@ wall_clock_limit = 48:15:50
# #@ job_name = eos-%d
# #@ initialdir = $(home)/workspace/eos-scripts/bayes2
# #@ output = %s
# #@ error  = %s
# #@ notification=error
# #@ notify_user=Frederik.Beaujean@lmu.de
# #@ queue

# %s
# """ % ((self.options.final_queue if final else self.options.queue),
#         self.step, log_file_name, log_file_name, script_name)

#         status, output = commands.getstatusoutput(cmd)

#         if status != 0 or not output:
#             raise Exception("Couldn't submit to LL queue: \n Tried '%s'\n\n and received:\n\n %s" % (cmd, output))

#         # parse job id from a line like
#         # llsubmit: The job "xcat.163140" has been submitted.
#         start = output.find('.')
#         end = output.find('"', start)
#         job_id = int(output[start + 1: end])

    def start_job(self, job_index, final=False):
        """
        Start a single job with the given index.
        Return the job id assigned by LL
        """
        log_file_name = self.output_job_base + '_%d' % job_index + '.log'

        self.clean_files.append(log_file_name)

        return self.submit(script_name=self.job_infos[job_index].script_name,
                    log_file_name=log_file_name,
                    queue=self.options.final_queue if final else self.options.queue,
                    job_name='job_%d_%d' % (self.step, job_index))

        # cmd =  'qsub'
        # cmd += ' -q %s' % (self.options.final_queue if final else self.options.queue)
        # cmd += ' -j y' # what does it mean?
        # cmd += ' -N job_%d_%d' % (self.step, job_index) #job_indentifier
        # cmd += ' -e %s' % log_file_name
        # cmd += ' -o %s' % log_file_name
        # cmd += ' -V'
        # cmd += ' -S /bin/bash'
        # cmd += ' %s' % self.job_infos[job_index].script_name

        # status, output = commands.getstatusoutput(cmd)

        # if status != 0 or not output:
        #     raise Exception("Couldn't submit to LL queue: \n Tried '%s'\n\n and received:\n\n %s" % (cmd, output))

        # # remove file immediately, sometimes get this error when removing due to incredibly slow file system:
        # """OSError: [Errno 110] Connection timed out:"""
        # try:
        #     if os.path.exists(self.job_infos[job_index].script_name):
        #         pass
        #     #                os.remove(self.job_infos[job_index].script_name)
        # except OSError:
        #     pass

        # # parse job id from a line like
        # # 'Your job 2756454 ("empty_job.sh") has been submitted'
        # start = output.find('job')
        # end = output.find('("')
        # return int(output[start + 4: end])

def restart_jobs():
    """Submit jobs after an OSError has stopped program"""
    raise NotImplementedError()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Job manager for PMC')
    parser.add_argument('--first-step', help='Read info from merged prerun, and just initialize PMC', action='store_true')
    parser.add_argument('--final', help="""Set status to converged, and just perform final step.
                                         Requires input via --input""", action='store_true')
    parser.add_argument('--force-final-step', help='Perform final step, even if PMC did not converge.', action='store_true')
    parser.add_argument('--input', help='HDF5 input file name providing the propopsal density', action='store')
    parser.add_argument('--merge', help='Merge individual job files. Use with --step and --input.', action='store_true')
    parser.add_argument('--n-samples', help='Use only reduced number of samples for uncertainty put(testing)', action='store')
    parser.add_argument('--resume-update', help='Read info from merge and perform update, then continue PMC sampling', action='store_true')
    parser.add_argument('--resume-samples', help='Resuming sampling. Assume step is properly defined, filename constructed automatically',
                        action='store_true')
    parser.add_argument('--restart', help="restart jobs to queue", action='store_true')
    parser.add_argument('--step', help='Set step', action='store')
    parser.add_argument('--uncertainty-propagation', help='Run uncertainty propagation on posterior samples', action='store_true')
    parser.add_argument('--resource-manager', help='Resource manager selection for your cluster: SGE,Slurm, loadleveler', default='SGE')
    args = parser.parse_args()
    print("Initializing with args:")
    print(args.__dict__)
    if args.resource_manager == "SGE":
        pmc_iterator = SGE_Iterator()
    elif args.resource_manager == "Slurm":
        pmc_iterator = Slurm_Iterator()
    elif args.resource_manager == 'loadleveler':
        pmc_iterator = LL_Iterator()
    else:
        raise Exception("Invalid resource manager: %s" % args.resource_manager)

    if args.__dict__['uncertainty_propagation']:
        pmc_iterator.run_uncertainty(args.n_samples)
    elif args.restart:
        restart_jobs()
    else:
        pmc_iterator.run(args.__dict__['first_step'], args.__dict__['resume_update'],
                         args.__dict__['resume_samples'], bool(args.final),
                         int(args.step) if args.step else None, args.input,
                         force_final=bool(args.force_final_step), merge=args.merge)

if __name__ == '__main__':
    main()
