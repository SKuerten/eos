#vim: set sts=4 et :

echo "[scan loaded]"

export SCAN_scI="
        --scan      Re{c7}      -2.0     +2.0              --prior flat
        --scan      Re{c9}     -15.0    +15.0              --prior flat
        --scan      Re{c10}    -15.0    +15.0              --prior flat
"

export SCAN_scII="
        --scan      Re{c7}      -2.0     +2.0              --prior flat
        --scan      Im{c7}      -2.0     +2.0              --prior flat
        --scan      Re{c9}     -15.0    +15.0              --prior flat
        --scan      Im{c9}     -15.0    +15.0              --prior flat
        --scan      Re{c10}    -15.0    +15.0              --prior flat
        --scan      Im{c10}    -15.0    +15.0              --prior flat
"

export SCAN_scIII="
        --scan      Re{c7}      -1.0     +1.0              --prior flat
        --scan      Re{c9}      -7.5     +7.5              --prior flat
        --scan      Re{c10}     -7.5     +7.5              --prior flat
        --scan      Re{c7'}     -1.0     +1.0              --prior flat
        --scan      Re{c9'}     -7.5     +7.5              --prior flat
        --scan      Re{c10'}    -7.5     +7.5              --prior flat
"

export SCAN_sm=""
