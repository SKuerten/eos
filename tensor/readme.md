Uncertainty propagation
========================

define scenario and observable
---------------------------

1. Open `unc_observable.bash`, add one variable for the observable
   with options and kinematics.

2. Open `unc_scenario.bash`, add the *fixed* parameter to vary in a range, and
   define other parameters with non-default values.

prepare input
-------------

1. Run multiple MCMCs to create parameter samples
   `BASE_NAME=$WORK/eos/2015-tensor/2015-03-25 ./loadleveler-single.cmd sm-unc_K.bash mcmc 1 100`

2. merge the samples, remove burn-in and thin out `merge.py

   ```
   cd $BASE_NAME/sm_unc_K
   merge.py --pypmc --skip-init 0.05 --thin 100 --single-chain
   ```

3. plot to check for example correlation of samples
   `plotScript.py mcmc_pre_merged.hdf5 --pypmc --mcmc --nuis`

compute
-------

Each job calls `eos-evaluate` `N` times to compute *one* observable for fixed
parameter values. In the `i`-th call,
* the nuisance parameters are taken from the i-th MCMC sample,
* the observable is evaluated at `K` values of the *fixed* parameter,
* extra parameters (e.g. `Re{c9}`) are set to a non-standard value that does not change with `i`.

1. Test a small job on 5 parameter samples with scenario `ct`, observable `K_FH1to6`, input file, and sample range
   ```
   BASE_NAME=$WORK/eos/2015-tensor/2015-04-01 time ./unc_job.bash ct K_FH1to6 $BASE_NAME/sm_unc_K/mcmc_pre_merged.hdf5 0 5
   ```

2. Run 100 jobs over full range of samples
   `BASE_NAME=$WORK/eos/2015-tensor/2015-04-01 ./loadleveler-unc.cmd ct K_FH1to6 /gpfs/work/pr85tu/ru72xaf2/eos/2015-tensor/2015-04-01/sm_unc_K/mcmc_pre_merged.hdf5 100`

3. Merge outputs from parallel run
   `merge.py --unc --pypmc`

4. Check if all is well
   `plotScript.py unc_merged.hdf5 --pypmc --1D-bins 30`
