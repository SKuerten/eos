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

### MCMC

1. Run multiple MCMCs to create parameter samples
   `BASE_NAME=$WORK/eos/2015-tensor/2015-03-25 ./loadleveler-single.cmd sm-unc_K.bash mcmc 1 100`

2. merge the samples, remove burn-in and thin out `merge.py

   ```
   cd $BASE_NAME/sm_unc_K
   merge.py --pypmc --skip-init 0.05 --thin 100 --single-chain
   ```

3. plot to check for example correlation of samples
   `plotScript.py mcmc_pre_merged.hdf5 --pypmc --mcmc --nuis`

### importance sampling

Create file `vb.hdf5`

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

   Run 50 jobs for all observables
   ```bash
   export BASE_NAME=$WORK/eos/2015-tensor/2015-05-19
   input_file=/gpfs/work/pr85tu/ru72xaf2/eos/2015-tensor/2015-05-19/sm_unc_Kstar/mcmc_pre_merged.hdf5
   observables="BR1to6 BR14to16 BR16to19 J_1c_plus_J_2c1to6 J_1c_plus_J_2c15to19 J_1s_minus_3J_2s1to6 J_1s_minus_3J_2s15to19"
   for obs in $observables; do ./loadleveler-unc.cmd ct K_star_${obs} ${input_file} 50; done
   ```

3. Merge outputs from parallel run
   `merge.py --unc --pypmc`

    `for obs in $observables; do cd ct_Kstar_${obs} && merge.py --pypmc --unc && cd ..; done`

4. Check if all is well
   `plotScript.py unc_merged.hdf5 --pypmc --1D-bins 30`

5. copy over
    ```
    # on desktop machine
    for obs in $observables; do rsync -avR c2pap:/gpfs/work/pr85tu/ru72xaf2/eos/2015-tensor/2015-05-19/ct_c9_1dot1_Kstar_${obs}/unc_merged.hdf5 ./; done

    Kobs=$(ls 2015-04-01/)
    for obs in $Kobs; do ln -s 2015-04-01/${obs}/unc_merged.hdf5 unc_${obs}.hdf5; done
    ```

MCMC
====

compute
-------

`export BASE_NAME=$WORK/eos/2015-tensor/2015-05-19`

1. Single test run

    ```bash
    export EOS_ANALYSIS_INFO=1 # output constraints, parameters etc. to log file
    ./scTT5-K_KstarBR.bash mcmc 0139
    ```

2. 20 chains with random seed offset 1..20
   `./loadleveler-single.cmd ./scTT5-K_KstarBR.bash mcmc 1 20`

analyze
-------

1. check for test runs in output directory
   ```
   cd scTT5_K_KstarBR
   ls
   rm mcmc_01* # test after 13:00
   ```
2. merge, check output for chains with very low mode compared to best chains
   ```
   merge.py --pypmc
   plotScript.py mcmc_pre_merged.hdf5 --mcmc --pypmc --skip-init 0.2
   ```
