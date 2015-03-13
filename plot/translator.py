def wilson(i):
    return '\\mathcal{C}_{%s}' % str(i)

def subleading(transversity, helicity, particle=r'K^{\ast}'):
    return r"$\zeta_{%s}^{%s_{%s}}$" % (particle, helicity, transversity)

class EOS_Translator:
    """
    Translate EOS tex_map into LaTex tex_map.
    """

    tex_map = {# scan parameters
                  "Abs{c7}" :r"$\left|%s\right|$" % wilson(7),
                  "Abs{c9}" :r"$\left|%s\right|$" % wilson(9) ,
                  "Abs{c10}":r"$\left|%s\right|$" % wilson(10) ,
                  "Arg{c7}" :r"$\arg \, %s$" % wilson(7) ,
                  "Arg{c9}" :r"$\arg \, %s$" % wilson(9) ,
                  "Arg{c10}":r"$\arg \, %s$" % wilson(10) ,
                  "Abs{c7'}" :r"$\left|%s^{\prime}\right|$" % wilson(7) ,
                  "Abs{c9'}" :r"$\left|%s^{\prime}\right|$" % wilson(9) ,
                  "Abs{c10'}":r"$\left|%s^{\prime}\right|$" % wilson(10) ,
                  "Arg{c7'}" :r"$\arg \, %s^{\prime}$" % wilson(7) ,
                  "Arg{c9'}" :r"$\arg \, %s^{\prime}$" % wilson(9) ,
                  "Arg{c10'}":r"$\arg \, %s^{\prime}$" % wilson(10) ,
                  "Re{c7}":"$%s$" % wilson(7),
                  "Re{c9}":"$%s$" % wilson(9),
                  "Re{c10}":"$%s$"% wilson(10),
                  "Re{c7'}":"$%s^{\prime}$" % wilson(7),
                  "Re{c9'}":"$%s^{\prime}$" % wilson(9),
                  "Re{c10'}":"$%s^{\prime}$"% wilson(10),
                  "Im{c7}":r"$\Im\left(%s\right)$" % wilson(7) ,
                  "Im{c9}":r"$\Im\left(%s\right)$" % wilson(9) ,
                  "Im{c10}":r"$\Im\left(%s\right)$" % wilson(10) ,
                  "Im{c7'}":r"$\Im\left(%s^{\prime}\right)$" % wilson(7) ,
                  "Im{c9'}":r"$\Im\left(%s^{\prime}\right)$" % wilson(9) ,
                  "Im{c10'}":r"$\Im\left(%s^{\prime}\right)$" % wilson(10) ,
                  "Re{cS}":"$%s$" % wilson('S'),
                  "Re{cP}":"$%s$" % wilson('P'),
                  "Re{cT}":"$%s$"% wilson('T'),
                  "Re{cT5}":"$%s$"% wilson('T5'),
                  "Im{cS}":r"$\Im\left(%s\right)$" % wilson('S') ,
                  "Im{cP}":r"$\Im\left(%s\right)$" % wilson('P') ,
                  "Im{cT}":r"$\Im\left(%s\right)$" % wilson('T') ,
                  "Im{cT5}":r"$\Im\left(%s\right)$" % wilson('T5') ,
                  "Abs{cS}":r"$\left|%s\right|$" % wilson('S'),
                  "Abs{cP}":r"$\left|%s\right|$" % wilson('P'),
                  "Abs{cT}":r"$\left|%s\right|$" % wilson('T'),
                  "Abs{cT5}":r"$\left|%s\right|$"% wilson('T5'),
                  "Arg{cS}":r"$\arg \, %s$" % wilson('S'),
                  "Arg{cP}":r"$\arg \, %s$" % wilson('P'),
                  "Arg{cT}":r"$\arg \, %s$"% wilson('T'),
                  "Arg{cT5}":r"$\arg \, %s$"% wilson('T5'),
                  "Re{cS'}":"$%s^{\prime}$" % wilson('S'),
                  "Re{cP'}":"$%s^{\prime}$" % wilson('P'),
                  "Im{cS'}":r"$\Im\left(%s^{\prime}\right)$" % wilson('S') ,
                  "Im{cP'}":r"$\Im\left(%s^{\prime}\right)$" % wilson('P') ,
                  "Abs{cS'}":r"$\left|%s^{\prime}\right|$" % wilson('S'),
                  "Abs{cP'}":r"$\left|%s^{\prime}\right|$" % wilson('P'),
                  "Arg{cS'}":r"$\arg \, %s^{\prime}$" % wilson('S'),
                  "Arg{cP'}":r"$\arg \, %s^{\prime}$" % wilson('P'),
                  # masses
                  "mass::b(MSbar)":r"$m_{b}$",
                  "mass::c":r"$m_{c}$",
                  # CKM
                  "CKM::rhobar":r"$\bar{\rho}_{CKM}$",
                  "CKM::etabar":r"$\bar{\eta}_{CKM}$",
                  "CKM::lambda":r"$\lambda_{CKM}$",
                  "CKM::A":r"$A_{CKM}$",
                  # hadronic
                  "decay-constant::B_d":r"$F_{B_d}$",
                  "decay-constant::B_s":r"$F_{B_s}$",
                  "decay-constant::K_d":r"$F_{K_d}$",
                  "decay-constant::K_u":r"$F_{K_u}$",
                  "B->K^*::f_Kstar_perp@2GeV":r"$F_{K^*_{\perp}}$",
                  "B->K^*::f_Kstar_par":r"$F_{K^*_{\parallel}}$",
                  "lambda_B_p":r"$\Lambda_{B_p}$",
                  # form factors
                  "B->K^*::a0_uncertainty@BZ2004":r"$FF(BZ2004):a_0$",
                  "B->K^*::a1_uncertainty@BZ2004":r"$FF(BZ2004):a_1$",
                  "B->K^*::a2_uncertainty@BZ2004":r"$FF(BZ2004):a_2$",
                  "B->K^*::v_uncertainty@BZ2004":r"$FF(BZ2004):v$",
                  "B->K::F^p(0)@KMPW2010":r"$f_+(0)$",
                  "B->K::F^t(0)@KMPW2010":r"$f_T(0)$",
                  "B->K::b^p_1@KMPW2010":r"$b^+_1$",
                  "B->K::b^0_1@KMPW2010":r"$b^0_1$",
                  "B->K::b^t_1@KMPW2010":r"$b^T_1$",
                  "B->K^*::F^V(0)@KMPW2010":r"$V(0)$",
                  "B->K^*::b^V_1@KMPW2010":r"$b^V_1$",
                  "B->K^*::F^A1(0)@KMPW2010":r"$A_1(0)$",
                  "B->K^*::b^A1_1@KMPW2010":r"$b^{A_1}_1$",
                  "B->K^*::F^A2(0)@KMPW2010":r"$A_2(0)$",
                  "B->K^*::b^A2_1@KMPW2010":r"$b^{A_2}_1$",
                  #subleading uncertainties
                  "B->Pll::Lambda_pseudo@LowRecoil":r"SL$(B \to P l^+ l^- \mathrm{@ low recoil}): \Lambda$",
                  "B->Pll::sl_phase_pseudo@LowRecoil":r"SL$(B \to P l^+ l^- \mathrm{@ low recoil}): \varphi$",
                  "B->Pll::Lambda_pseudo@LargeRecoil":r"SL$(B \to P l^+ l^- \mathrm{@ large recoil}): \Lambda$",
                  "B->Pll::sl_phase_pseudo@LargeRecoil":r"SL$(B \to P l^+ l^- \mathrm{@ large recoil}): \varphi$",
                  "B->Vll::Lambda_0@LowRecoil":r"SL$(B \to V l^+ l^- \mathrm{@ low recoil}): \Lambda_0$",
                  "B->Vll::Lambda_pa@LowRecoil":r"SL$(B \to V l^+ l^- \mathrm{@ low recoil}): \Lambda_{\parallel}$",
                  "B->Vll::Lambda_pp@LowRecoil":r"SL$(B \to V l^+ l^- \mathrm{@ low recoil}): \Lambda_{\perp}$",
                  "B->Vll::sl_phase_0@LowRecoil":r"SL$(B \to V l^+ l^- \mathrm{@ low recoil}): \varphi_0$",
                  "B->Vll::sl_phase_pa@LowRecoil":r"SL$(B \to V l^+ l^- \mathrm{@ low recoil}): \varphi_{\parallel}$",
                  "B->Vll::sl_phase_pp@LowRecoil":r"SL$(B \to V l^+ l^- \mathrm{@ low recoil}): \varphi_{\perp}$",
                  "B->K^*ll::A_0^L_uncertainty@LargeRecoil":subleading('0', 'L'),
                  "B->K^*ll::A_0^R_uncertainty@LargeRecoil":subleading('0', 'R'),
                  "B->K^*ll::A_par^L_uncertainty@LargeRecoil":subleading(r'\parallel', 'L'),
                  "B->K^*ll::A_par^R_uncertainty@LargeRecoil":subleading(r'\parallel', 'R'),
                  "B->K^*ll::A_perp^L_uncertainty@LargeRecoil":subleading(r'\perp', 'L'),
                  "B->K^*ll::A_perp^R_uncertainty@LargeRecoil":subleading(r'\perp', 'R'),
                  "B->Vll::Lambda@LowRecoil":r"SL$(B \to V l^+ l^- \mathrm{@ low recoil}): \Lambda_{\mathrm{simple}}$",
                  'B->Vll::sl_phase@LowRecoil':r"SL$(B \to V l^+ l^- \mathrm{@ low recoil}): \varphi_{\mathrm{simple}}$",
                  'B->K^*ll::sl_uncertainty@LargeRecoil':r"SL$(B \to V l^+ l^- \mathrm{@ large recoil}): \Lambda_{\mathrm{simple}}$",

                  # observables
                  "B_q->ll::BR":r"$\mathcal{B}(B_s \to \bar{\mu}\mu $",
                  "B->Kll::BR@LargeRecoil":r"$\mathcal{B}(B \to K \bar{\ell}\ell $",
                  "B->Kll::BR@LowRecoil":r"$\mathcal{B}(B \to K \bar{\ell}\ell $",
                  "B->K^*ll::BR@LargeRecoil":r"$ \langle \mathcal{B} \rangle $",
                  "B->K^*ll::BR@LowRecoil":r"$ \langle \mathcal{B} \rangle $",
                  "B->K^*ll::F_L@LargeRecoil":r"$ \langle F_L \rangle $",
                  "B->K^*ll::F_L@LowRecoil":r"$ \langle F_L \rangle $",
                  "B->K^*ll::A_FB@LargeRecoil":r"$ \langle A_{\rm FB} \rangle $",
                  "B->K^*ll::A_FB@LowRecoil":r"$ \langle A_{\rm FB} \rangle $",
                  "B->K^*ll::A_T^2@LargeRecoil":r"$A_T^{(2)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::A_T^2@LowRecoil":r"$A_T^{(2)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::A_T^3@LargeRecoil":r"$ \langle A_T^{(3)} \rangle $",
                  "B->K^*ll::A_T^3@LowRecoil":r"$A_T^{(3)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::A_T^4@LargeRecoil":r"$ \langle A_T^{(4)} \rangle $",
                  "B->K^*ll::A_T^4@LowRecoil":r"$A_T^{(4)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::A_T^5@LargeRecoil":r"$ \langle A_T^{(5)} \rangle $",
                  "B->K^*ll::A_T^5@LowRecoil":r"$A_T^{(5)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::A_T^re@LargeRecoil":r"$ \langle A_T^{(re)} \rangle $",
                  "B->K^*ll::A_T^re@LowRecoil":r"$A_T^{(re)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::A_T^im@LargeRecoil":r"$A_T^{(im)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::A_T^im@LowRecoil":r"$A_T^{(im)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::H_T^1@LargeRecoil":r"$ \langle H_T^{(1)} \rangle $",
                  "B->K^*ll::H_T^1@LowRecoil":r"$ \langle H_T^{(1)} \rangle $",
                  "B->K^*ll::H_T^2@LargeRecoil":r"$ \langle H_T^{(2)} \rangle $",
                  "B->K^*ll::H_T^2@LowRecoil":r"$H_T^{(2)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::H_T^3@LargeRecoil":r"$H_T^{(3)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::H_T^3@LowRecoil":r"$H_T^{(3)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::H_T^4@LargeRecoil":r"$H_T^{(4)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::H_T^4@LowRecoil":r"$H_T^{(4)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::H_T^5@LargeRecoil":r"$H_T^{(5)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::H_T^5@LowRecoil":r"$H_T^{(5)}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_1s@LargeRecoil":r"$J_{1s}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_1s@LowRecoil":r"$J_{1s}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_1c@LargeRecoil":r"$J_{1c}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_1c@LowRecoil":r"$J_{1c}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_2s@LargeRecoil":r"$J_{2s}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_2s@LowRecoil":r"$J_{2s}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_2c@LargeRecoil":r"$J_{2c}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_2c@LowRecoil":r"$J_{2c}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_3@LargeRecoil":r"$J_{3}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_3@LowRecoil":r"$J_{3}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_4@LargeRecoil":r"$J_{4}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_4@LowRecoil":r"$J_{4}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_5@LargeRecoil":r"$J_{5}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_5@LowRecoil":r"$J_{5}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_6s@LargeRecoil":r"$J_{6s}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_6s@LowRecoil":r"$J_{6s}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_6c@LargeRecoil":r"$J_{6c}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_6c@LowRecoil":r"$J_{6c}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_7@LargeRecoil":r"$J_{7}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_7@LowRecoil":r"$J_{7}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_8@LargeRecoil":r"$J_{8}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_8@LowRecoil":r"$J_{8}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_9@LargeRecoil":r"$J_{9}(B \to K^{*} \bar{\ell}\ell $",
                  "B->K^*ll::J_9@LowRecoil":r"$J_{9}(B \to K^{*} \bar{\ell}\ell $",

                  # constraints
                  "B^+->K^+mu^+mu^-::BR[1.00,6.00]":r"$\mathcal{B} [1,6]$",
                  "B^+->K^+mu^+mu^-::BR[14.18,16.00]":r"$\mathcal{B} [14.18,16]$",
                  "B^0->K^*0mu^+mu^-::BR[1.00,6.00]":r"$\mathcal{B} [1,6]$",
                  'B^0->K^*0mu^+mu^-::A_FB[1.00,6.00]':r"$A_{\mathrm{FB}} [1,6]$",
                  'B^0->K^*0mu^+mu^-::F_L[1.00,6.00]':r"$F_{L} [1,6]$",
                  'B^0->K^*0mu^+mu^-::S_3[1.00,6.00]':r"$S_{3} [1,6]$",
                  "B^0->K^*0mu^+mu^-::BR[14.18,16.00]":r"$\mathcal{B} [14.18,16]$",
                  'B^0->K^*0mu^+mu^-::A_FB[14.18,16.00]':r"$A_{FB} [14.18,16]$",
                  'B^0->K^*0mu^+mu^-::F_L[14.18,16.00]':r"$F_{L} [14.18,16]$",
                  'B^0->K^*0mu^+mu^-::S_3[14.18,16.00]':r"$S_{3} [14.18,16]$",
                  "B^0->K^*0mu^+mu^-::BR[16.00,19.21]":r"$\mathcal{B} [16,19.21]$",
                  'B^0->K^*0mu^+mu^-::A_FB[16.00,19.21]':r"$A_{FB} [16,19.21]$",
                  'B^0->K^*0mu^+mu^-::F_L[16.00,19.21]':r"$F_{L} [16,19.21]$",
                  'B^0->K^*0mu^+mu^-::F_L[16.00,19.00]':r"$F_{L} [16,19.00]$",
                  'B^0->K^*0mu^+mu^-::S_3[16.00,19.00]':r"$S_{3} [16,19.00]$",
                  'B^0->K^*0mu^+mu^-::A_T_2[1.00,6.00]':r"$A_T^{(2)} [1,6]$",
                  'B^0->K^*0mu^+mu^-::A_T_2[14.18,16.00]':r"$A_T^{(2)} [14.18,16]$",
                  'B^0->K^*0mu^+mu^-::A_T_2[16.00,19.21]':r"$A_T^{(2)} [16,19.21]$",
                  'B^0->K^*0gamma::BR':r"$\mathcal{B}$",
                  'B^0->K^*0gamma::S_K+C_K':r"$S + C$",
                  'B^0_s->mu^+mu^-::BR_limit':r"$\mathcal{B}(B_s \to \bar{\mu} \mu)$",

                  # experiments
                  "BaBar":r"$\mathrm{BaBar}$",
                  "Belle":r"$\mathrm{Belle}$",
                  "CLEO":r"$\mathrm{CLEO}$",
                  "CDF":r"$\mathrm{CDF}$",
                  "LHCb":r"$\mathrm{LHCb}$",
                  }

    def to_tex(self, name):
        "Translate name to a texname. Defaults to name if translation unknown"
        return self.tex_map.get(name, name)

def print_map():
    "Print rendered map in multiple columns"
    from matplotlib import pyplot as P
    xsize = 20
    fig = P.figure(figsize=(xsize, 3. / 4 * xsize))
    ax = P.gca()
    tr = EOS_Translator()
#     print(tr.to_tex("Arg{c10'}"))

    columns = ['', '', '']
    for i, key in enumerate(sorted(tr.tex_map.iterkeys())):
        col = i / (len(tr.tex_map) / len(columns) + 1)
        columns[col] += '%s: %s\n' % (key, tr.to_tex(key))

    for i, c in enumerate(columns):
        P.text(i * 1. / len(columns) + 0.01, 0.99, c,
               transform=fig.transFigure,
               verticalalignment='top')

    ax.set_axis_off()
    P.savefig('translator-map.pdf')

if __name__ == '__main__':
    print_map()
