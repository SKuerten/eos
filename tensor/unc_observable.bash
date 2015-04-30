GLOBAL_OPTIONS=",model=WilsonScan,scan-mode=cartesian"

K_OPTIONS=",q=u,l=mu,form-factors=KMPW2010"

export UNC_OBS_K_FH1to6="B->Kll::F_Havg@LargeRecoil${GLOBAL_OPTIONS}${K_OPTIONS}"
export UNC_KIN_K_FH1to6="s_min 1 s_max 6"

export UNC_OBS_K_FH15to22="B->Kll::F_Havg@LowRecoil${GLOBAL_OPTIONS}${K_OPTIONS}"
export UNC_KIN_K_FH15to22="s_min 15.00 s_max 22.00"

export UNC_OBS_K_BR1to6="B->Kll::BRavg@LargeRecoil${GLOBAL_OPTIONS}${K_OPTIONS}"
export UNC_KIN_K_BR1to6="s_min 1 s_max 6"

export UNC_OBS_K_BR15to22="B->Kll::BRavg@LowRecoil${GLOBAL_OPTIONS}${K_OPTIONS}"
export UNC_KIN_K_BR15to22="s_min 15.00 s_max 22.00"

KSTAR_OPTIONS=",q=d,l=mu,form-factors=BSZ2015"

export UNC_OBS_Kstar_BR1to6="B->K^*ll::BRavg@LargeRecoil${GLOBAL_OPTIONS}${KSTAR_OPTIONS}"
export UNC_KIN_Kstar_BR1to6="s_min 1 s_max 6"

export UNC_OBS_Kstar_BR14to16="B->K^*ll::BRavg@LowRecoil${GLOBAL_OPTIONS}${KSTAR_OPTIONS}"
export UNC_KIN_Kstar_BR14to16="s_min 14.18 s_max 16"

export UNC_OBS_Kstar_BR16to19="B->K^*ll::BRavg@LowRecoil${GLOBAL_OPTIONS}${KSTAR_OPTIONS}"
export UNC_KIN_Kstar_BR16to19="s_min 16 s_max 19"


export UNC_OBS_Kstar_J_1c_plus_J_2c1to6="B->K^*ll::J_1c+J_2cavg@LargeRecoil${GLOBAL_OPTIONS}${KSTAR_OPTIONS}"
export UNC_KIN_Kstar_J_1c_plus_J_2c1to6=${UNC_KIN_Kstar_BR1to6}

export UNC_OBS_Kstar_J_1c_plus_J_2c15to19="B->K^*ll::J_1c+J_2cavg@LowRecoil${GLOBAL_OPTIONS}${KSTAR_OPTIONS}"
export UNC_KIN_Kstar_J_1c_plus_J_2c15to19="s_min 15 s_max 19"


export UNC_OBS_Kstar_J_1s_minus_3J_2s1to6="B->K^*ll::J_1s-3J_2savg@LargeRecoil${GLOBAL_OPTIONS}${KSTAR_OPTIONS}"
export UNC_KIN_Kstar_J_1s_minus_3J_2s1to6=${UNC_KIN_Kstar_BR1to6}

export UNC_OBS_Kstar_J_1s_minus_3J_2s15to19="B->K^*ll::J_1s-3J_2savg@LowRecoil${GLOBAL_OPTIONS}${KSTAR_OPTIONS}"
export UNC_KIN_Kstar_J_1s_minus_3J_2s15to19=${UNC_KIN_Kstar_J_1c_plus_J_2c15to19}
