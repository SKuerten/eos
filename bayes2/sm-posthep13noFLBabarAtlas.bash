#!/bin/bash
#vim: set sts=4 et :

source ${EOS_SCRIPT_PATH}/job.bash

# initial value from sm best fit
export GOF_MODE_0="{
 +0.8118397518535488 +0.2255445464785774 +0.1100037939503916
 +0.3703299055637777 +1.2744132117575957 +4.1909748234720361
 +0.2289861184520069 +0.3856052030666717 -4.6923706819401714
 +0.8442200868320692 +1.1175613752050411 +0.2417248844208102
 +0.2324725096072476 +0.2274420476998674 -0.6824727437358048
 -0.0158954180001218 -0.0097150103440470 -0.0129892315052538
 +0.8816621876363251 +1.0348386995119681 +1.1668183018335545
 +1.0552634546381101 +0.2961043135325561 -3.1506199403720880
 -0.0223635913944357 -0.3370481089296705 +0.4265276232485582
 +0.3503791385760284 }"

main $@
