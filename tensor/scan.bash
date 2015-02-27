echo "[scan loaded]"

sm_min=-7
sm_max=+7
non_sm_min=-1
non_sm_max=+1
sm_min=-7
sm_max=+7

export SCAN_scTT5="
--global-option model WilsonScan
--global-option scan-mode cartesian

--scan      Re{cT}      $non_sm_min     $non_sm_max  --prior flat
--scan      Im{cT}      $non_sm_min     $non_sm_max  --prior flat
--scan      Re{cT5}     $non_sm_min     $non_sm_max  --prior flat
--scan      Im{cT5}     $non_sm_min     $non_sm_max  --prior flat
"

export SCAN_sc910TT5="
--global-option model WilsonScan
--global-option scan-mode cartesian

--scan      Re{c9}      $sm_min         $sm_max      --prior flat
--scan      Im{c9}      $sm_min         $sm_max      --prior flat
--scan      Re{c10}     $sm_min         $sm_max      --prior flat
--scan      Im{c10}     $sm_min         $sm_max      --prior flat
--scan      Re{cT}      $non_sm_min     $non_sm_max  --prior flat
--scan      Im{cT}      $non_sm_min     $non_sm_max  --prior flat
--scan      Re{cT5}     $non_sm_min     $non_sm_max  --prior flat
--scan      Im{cT5}     $non_sm_min     $non_sm_max  --prior flat
"

export SCAN_scSP="
--global-option model WilsonScan
--global-option scan-mode cartesian

--scan      Re{cS}      $non_sm_min     $non_sm_max  --prior flat
--scan      Im{cS}      $non_sm_min     $non_sm_max  --prior flat
--scan      Re{cS'}     $non_sm_min     $non_sm_max  --prior flat
--scan      Im{cS'}     $non_sm_min     $non_sm_max  --prior flat
--scan      Re{cP}      $non_sm_min     $non_sm_max  --prior flat
--scan      Im{cP}      $non_sm_min     $non_sm_max  --prior flat
--scan      Re{cP'}     $non_sm_min     $non_sm_max  --prior flat
--scan      Im{cP'}     $non_sm_min     $non_sm_max  --prior flat
"
