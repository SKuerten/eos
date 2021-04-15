/* vim: set sw=4 sts=4 et foldmethod=syntax : */

/*
 * Copyright (c) 2020 Danny van Dyk
 * Copyright (c) 2020 Stephan Kuerten
 *
 * This file is part of the EOS project. EOS is free software;
 * you can redistribute it and/or modify it under the terms of the GNU General
 * Public License version 2, as published by the Free Software Foundation.
 *
 * EOS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <eos/b-decays/b-to-3l-nu.hh>
#include <eos/form-factors/mesonic.hh>
#include <eos/utils/destringify.hh>
#include <eos/utils/model.hh>
#include <eos/utils/power_of.hh>
#include <eos/utils/private_implementation_pattern-impl.hh>
#include <eos/utils/options-impl.hh>
#include <eos/utils/kinematic.hh>

namespace eos
{
    using std::norm;
    using std::real;
    using std::imag;
    using std::sqrt;

    /*
     * Decay: B_q^- -> l1^+ l1^- l2^- nubar, cf. [KKvD:2021A]
     */
    template <>
    struct Implementation<BToThreeLeptonsNeutrino>
    {
        std::shared_ptr<Model> model;

        std::shared_ptr<FormFactors<PToGammaOffShell>> form_factors;

        UsedParameter hbar;

        UsedParameter g_fermi;

        UsedParameter m_B;

        UsedParameter f_B;

        UsedParameter tau_B;

        UsedParameter alpha_qed;

        SwitchOption opt_l1;

        UsedParameter m_l1;

        SwitchOption opt_l2;

        UsedParameter m_l2;

        Implementation(const Parameters & p, const Options & o, ParameterUser & u) :
            model(Model::make(o.get("model", "SM"), p, o)),
            form_factors(FormFactorFactory<PToGammaOffShell>::create("B->gamma^*::" + o.get("form-factors", "KKvD2021"), p, o)),
            hbar(p["hbar"], u),
            g_fermi(p["G_Fermi"], u),
            m_B(p["mass::B_u"], u),
            f_B(p["decay-constant::B_u"], u),
            tau_B(p["life_time::B_u"], u),
            alpha_qed(p["QED::alpha_e(m_b)"],u),
            opt_l1(o, "l1", {"e", "mu"}, "mu"),
            m_l1(p["mass::" + opt_l1.value()], u),
            opt_l2(o, "l2", {"e", "mu"}, "e"),
            m_l2(p["mass::" + opt_l2.value()], u)
        {
            u.uses(*model);
            u.uses(*form_factors);
        }

        double decay_width(const double & q2, const double & k2) const
        {
            const WilsonCoefficients<ChargedCurrent> wc = model->wilson_coefficients_b_to_u(opt_l2.value(), false);

            const complex<double> cVL = wc.cvl();
            const complex<double> cVR = wc.cvr();

            const complex<double> F_perp = form_factors->F_perp(q2, k2);
            const complex<double> F_para = form_factors->F_para(q2, k2);
            const complex<double> F_long = form_factors->F_long(q2, k2);

            return power_of<2>(g_fermi * abs(model->ckm_ub() ) * abs(cVL) ) * alpha_qed *  (32.0 * M_PI)
                    * sqrt(lambda(q2, 0.0, m_l2 * m_l2) * lambda(k2, m_l1 * m_l1, m_l1 * m_l1) * lambda(m_B * m_B, q2, k2) )
                    * (norm(F_perp) + norm(F_para) + norm(F_long) );
        }

        double decay_width_5diff(const double & q2, const double & k2, const double & y/*cos(theta_W)*/, const double & z/*cos(theta_V)*/, const double & phi) const
        {
            const WilsonCoefficients<ChargedCurrent> wc = model->wilson_coefficients_b_to_u(opt_l2.value(), false);

            const complex<double> cVL = wc.cvl();
            const complex<double> cVR = wc.cvr();

            const complex<double> F_perp = 1.0;//form_factors->F_perp(q2, k2);
            const complex<double> F_para = 1.0;//form_factors->F_para(q2, k2);
            const complex<double> F_long = 1.0;//form_factors->F_long(q2, k2);

            const double cos_2phi =   cos(2.0 * phi);
            const double sin_2phi =   sin(2.0 * phi);
            const double cos_phi  =         cos(phi);
            const double sin_phi  =         sin(phi);

            // angular coefficients S_(l1,l2)^m are denoted as s_l1_l2_m, with 0 <= l1, l2 <= 2 and -2 <= m <= +2.
            const double s_0_0_p0  =   (norm(F_long) + norm(F_para) + norm(F_perp) ) / (8. * M_PI);
            const double s_0_1_p0  = -  sqrt(3.) * real(conj(F_perp) * F_para) / (4. * M_PI);
            const double s_0_2_p0  = - (norm(F_long) - 2. * norm(F_para) ) / (8. * M_PI);
            const double s_1_0_p0  =    0.0;
            const double s_1_1_m1  =    0.0;
            const double s_1_1_p0  =    0.0;
            const double s_1_1_p1  =    0.0;
            const double s_1_2_m1  =    0.0;
            const double s_1_2_p0  =    0.0;
            const double s_1_2_p1  =    0.0;
            const double s_2_0_p0  = - (norm(F_long) - 2. * norm(F_para) ) / (8. * M_PI);
            const double s_2_1_m1  = -  3. * sqrt(3.) * imag(conj(F_para) * F_long) / (8. * M_PI);
            const double s_2_1_p0  = -  sqrt(3.) * real(conj(F_perp) * F_para) / (8. * M_PI);
            const double s_2_1_p1  =    3. * real(conj(F_perp) * F_long) / (8. * M_PI);
            const double s_2_2_m2  = -  sqrt(3.) * imag(conj(F_perp) * F_para) / (4. * M_PI);
            const double s_2_2_m1  =    sqrt(3.) * imag(conj(F_perp) * F_long) / (8. * M_PI);
            const double s_2_2_p0  =   (2. * norm(F_long) - norm(F_para) + norm(F_perp) ) / (16. * M_PI);
            const double s_2_2_p1  = -  3. * real(conj(F_para) * F_long) / (8. * M_PI);
            const double s_2_2_p2  =   (3. * norm(F_para) - norm(F_perp) ) / (8. * M_PI);


            return power_of<2>(g_fermi * abs(model->ckm_ub() ) * abs(cVL) ) * alpha_qed *  (32.0 * M_PI)
                    * sqrt(lambda(q2, 0.0, m_l2 * m_l2) * lambda(k2, m_l1 * m_l1, m_l1 * m_l1) * lambda(m_B * m_B, q2, k2) ) * (
                    + s_0_0_p0
                    + s_0_1_p0  * y
                    + s_0_2_p0  * (-1. + 3. * y * y) / 2.
                    + s_1_0_p0  * z
                    + s_1_1_m1  * sqrt( (1. - y * y) * (1. - z * z) ) * sin_phi / 2.
                    + s_1_1_p0  * y * z
                    + s_1_1_p1  * sqrt( (1. - y * y) * (1. - z * z) ) * cos_phi / 2.
                    + s_1_2_m1  * y * sqrt(3. *(1. - y * y) * (1. - z * z) ) * sin_phi / 2.
                    + s_1_2_p0  * z * (-1. + 3. * y * y) / 2.
                    + s_1_2_p1  * y * sqrt(3. *(1. - y * y) * (1. - z * z) ) * cos_phi / 2.
                    + s_2_0_p0  * (-1. + 3. * z * z) / 2.
                    + s_2_1_m1  * z * sqrt(3. *(1. - y * y) * (1. - z * z) ) * sin_phi / 2.
                    + s_2_1_p0  * y * (-1. + 3. * z * z) / 2.
                    + s_2_1_p1  * z * sqrt(3. *(1. - y * y) * (1. - z * z) ) * cos_phi / 2.
                    + s_2_2_m2  * z * 3. * (-1. + y * y) * (-1. + z * z) * sin_2phi / 8.
                    + s_2_2_m1  * y * z * 3. * sqrt( (1. - y * y) * (1. - z * z) ) * sin_phi / 2.
                    + s_2_2_p0  * (-1. + 3. * y * y) * (-1. + 3. * z * z) / 4.
                    + s_2_2_p1  * y * z * 3. * sqrt( (1. - y * y) * (1. - z * z) ) * cos_phi / 2.
                    + s_2_2_p2  * y * 3. * (-1. + y * y) * (-1. + z * z) * cos_2phi / 8.
                    );
        }

        double branching_ratio(const double & q2, const double & k2) const
        {
            return decay_width(q2, k2) * tau_B / hbar;
        }

        double branching_ratio_5diff(const double & q2, const double & k2, const double & y/*cos(theta_W)*/, const double & z/*cos(theta_V)*/, const double & phi) const
        {
            const WilsonCoefficients<ChargedCurrent> wc = model->wilson_coefficients_b_to_u(opt_l2.value(), false);

            const complex<double> cVL = wc.cvl();
            return decay_width_5diff(q2, k2, y, z, phi) * tau_B / hbar;
        }
    };

    BToThreeLeptonsNeutrino::BToThreeLeptonsNeutrino(const Parameters & parameters, const Options & options) :
        PrivateImplementationPattern<BToThreeLeptonsNeutrino>(new Implementation<BToThreeLeptonsNeutrino>(parameters, options, *this))
    {
    }

    BToThreeLeptonsNeutrino::~BToThreeLeptonsNeutrino()
    {
    }

    double
    BToThreeLeptonsNeutrino::branching_ratio(const double & q2, const double & k2) const
    {
        return _imp->branching_ratio(q2, k2);
    }

    double
    BToThreeLeptonsNeutrino::branching_ratio_5diff(const double & q2, const double & k2, const double & y/*cos(theta_W)*/, const double & z/*cos(theta_V)*/, const double & phi) const
    {
        return _imp->branching_ratio_5diff(q2, k2, y, z, phi);
    }

    double
    BToThreeLeptonsNeutrino::decay_width(const double & q2, const double & k2) const
    {
        return _imp->decay_width(q2, k2);
    }

    double
    BToThreeLeptonsNeutrino::decay_width_5diff(const double & q2, const double & k2, const double & y/*cos(theta_W)*/, const double & z/*cos(theta_V)*/, const double & phi) const
    {
        return _imp->decay_width_5diff(q2, k2, y, z, phi);
    }

    const std::set<ReferenceName>
    BToThreeLeptonsNeutrino::references
    {
        "KKvD:2021A"_rn
    };
}
