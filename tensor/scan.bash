echo "[scan loaded]"

export SCAN_scTT5="
 --global-option model WilsonScan
 --global-option scan-mode cartesian

 --scan      Re{cT}      -1     +1  --prior flat
 --scan      Re{cT5}     -1     +1  --prior flat
"
