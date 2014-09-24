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

# BOBYQA
export GOF_MODE_0="{
 +0.8034386365391224 +0.2254042186844880 +0.1291984972886490
 +0.3796668098969205 +1.2741320216304999 +4.1926610224013361
 +0.2281245393633325 +0.3695077215528563 -4.8785783075715665
 +0.8792423929326187 +1.0632197437893631 +0.2430400865772986
 +0.2379088228482953 +0.2180038538122085 -0.6188485854597948
 +0.0103572176653451 -0.0143539280519032 +0.0011446549154731
 +0.9428541213385540 +1.0161024502385885 +1.0030922401451896
 +1.0018382237407630 +0.2988963718628981 -3.1253174640459442
 -0.0241747111121667 -0.3590449908789709 +0.4400009519965754
 +0.3505705877632184 }"

main $@
