#vim: set sts=4 et :

echo "[scan loaded]"

export SCAN_scI="
        --scan      Re{c7}      -2.0     +2.0              --prior flat
        --scan      Re{c9}     -15.0    +15.0              --prior flat
        --scan      Re{c10}    -15.0    +15.0              --prior flat
"

export SCAN_scII="
        --scan      Re{c9}      -7.5     +7.5              --prior flat
        --scan      Re{c9'}     -7.5     +7.5              --prior flat
"

export SCAN_scIII="
        --scan      Re{c7}      -1.0     +1.0              --prior flat
        --scan      Re{c9}      -7.5     +7.5              --prior flat
        --scan      Re{c10}     -7.5     +7.5              --prior flat
        --scan      Re{c7'}     -1.0     +1.0              --prior flat
        --scan      Re{c9'}     -7.5     +7.5              --prior flat
        --scan      Re{c10'}    -7.5     +7.5              --prior flat
"

export SCAN_scIIIA="
        --scan      Re{c7}      -1.0      0.0              --prior flat
        --scan      Re{c9}       0.0     +7.5              --prior flat
        --scan      Re{c10}     -7.5      0.0              --prior flat
        --scan      Re{c7'}     -0.4     +0.4              --prior flat
        --scan      Re{c9'}     -3.0     +5.0              --prior flat
        --scan      Re{c10'}    -3.0     +3.0              --prior flat
"

export SCAN_sm=""
