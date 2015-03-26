Uncertainty propagation
========================

1. Run multiple MCMCs to create parameter samples
   `BASE_NAME=$WORK/eos/2015-tensor/2015-03-25 ./loadleveler-single.cmd sm-unc_K.bash mcmc 1 100`

2. merge the samples, remove burn-in and thin out `merge.py

   ```
   cd $BASE_NAME/sm_unc_K
   merge.py --pypmc --cut-off 32 --skip-init 0.05 --thin 100 --single-chain
   ```

3. All jobs compute *one* observable for values of *one* parameter
   (the *fixed* parameter) in a fixed range and a set of samples from
   MCMC that define the other parameter values.

   Test a small job on 5 parameter samples
   `BASE_NAME=$WORK/eos/2015-tensor/2015-03-25 time ./sm-FH.bash unc 500 505`

   Run with 20 jobs over full range of samples
   `BASE_NAME=$WORK/eos/2015-tensor/2015-03-25 ./loadleveler-unc.cmd sm-Bsmumu.bash 20`

4. Merge outputs from parallel run
   `~/data/2015-03-25/sm_Bsmumu$ merge.py --unc --pypmc`
