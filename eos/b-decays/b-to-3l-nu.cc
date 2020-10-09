/* vim: set sw=4 sts=4 et foldmethod=syntax : */

/*
 * Copyright (c) 2020 Danny van Dyk
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

            // TODO(SK) -> implement the q^2 and k^2 differential branching
            // fraction
            return power_of<2>(g_fermi * abs(model->ckm_ub()) * abs(cVL)) * alpha_qed *  (32.0 * M_PI)
                    * sqrt(lambda(q2, 0.0, m_l2 * m_l2) * lambda(k2, m_l1 * m_l1, m_l1 * m_l1) * lambda(m_B * m_B, q2, k2))
                    * (norm(F_perp) + norm(F_para) + norm(F_long));
        }

        // TODO(SK) -> implement the 5 (q^2,k^2,cos(theta_W), cos(theta_V) and phi) differential branching
        double decay_width_5diff(const double & q2, const double & k2, const double & y/*cos(theta_W)*/, const double & z/*cos(theta_V)*/, const double & phi) const
        {
            const WilsonCoefficients<ChargedCurrent> wc = model->wilson_coefficients_b_to_u(opt_l2.value(), false);

            const complex<double> cVL = wc.cvl();
            const complex<double> cVR = wc.cvr();

            const complex<double> F_perp = form_factors->F_perp(q2, k2);
            const complex<double> F_para = form_factors->F_para(q2, k2);
            const complex<double> F_long = form_factors->F_long(q2, k2);

            const double cos_2phi =   cos(2.0 * phi);
            const double sin_2phi =   sin(2.0 * phi);
            const double cos_phi  =         cos(phi);
            const double sin_phi  =         sin(phi);

            const double s000  =   (7. * norm(F_long) + 3. * norm(F_para) + 5. * norm(F_perp)) / (32. * M_PI);
            const double s010  = - (3. * sqrt(3.) * real(conj(F_perp) * F_para)) / (16. * M_PI);
            const double s020  = - (4. * norm(F_long) - 9. * norm(F_para) + norm(F_perp)) / (64. * M_PI);
            const double s100  = - (3. * (3. * norm(F_long) + norm(F_para) + norm(F_perp))) / (32. * M_PI);
            const double s11m1 =   (9. *  imag(conj(F_para) * F_long)) / (25. * sqrt(2) * M_PI);
            const double s110  = - (3. * sqrt(3.) * real(conj(F_perp) * F_para)) / (16. * M_PI);
            const double s111  = - (3. * sqrt(3. / 2.) * real(conj(F_perp) * F_long)) / (25. * M_PI);
            const double s12m1 = - (4. * sqrt(2.) * imag(conj(F_perp) * F_long)) / (49. * M_PI);
            const double s120  = 1.0;
            const double s121  = 1.0;
            const double s200  = 1.0;
            const double s21m1 = 1.0;
            const double s210  = 1.0;
            const double s211  = 1.0;
            const double s22m2 = 1.0;
            const double s22m1 = 1.0;
            const double s220  = 1.0;
            const double s221  = 1.0;
            const double s222  = 1.0;


            return power_of<2>(g_fermi * abs(model->ckm_ub()) * abs(cVL)) * alpha_qed *  (32.0 * M_PI)
                    * sqrt(lambda(q2, 0.0, m_l2 * m_l2) * lambda(k2, m_l1 * m_l1, m_l1 * m_l1) * lambda(m_B * m_B, q2, k2)) * (
                    + s000
                    + s010  * y
                    + s020  * (-1. + 3. * y * y) / 2.
                    + s100  * z
                    + s11m1 * sqrt((1. - y * y) * (1. - z * z)) * sin_phi / 2.
                    + s110  * y * z
                    + s111  * sqrt((1. - y * y) * (1. - z * z)) * cos_phi / 2.
                    + s12m1 * y * sqrt(3. *(1. - y * y) * (1. - z * z)) * sin_phi / 2.
                    + s120  * z * (-1. + 3. * y * y) / 2.
                    + s121  * y * sqrt(3. *(1. - y * y) * (1. - z * z)) * cos_phi / 2.
                    + s200  * (-1. + 3. * z * z) / 2.
                    + s21m1 * z * sqrt(3. *(1. - y * y) * (1. - z * z)) * sin_phi / 2.
                    + s210  * y * (-1. + 3. * z * z) / 2.
                    + s211  * z * sqrt(3. *(1. - y * y) * (1. - z * z)) * cos_phi / 2.
                    + s22m2 * z * 3. * (-1. + y * y) * (-1. + z * z) * sin_2phi / 8.
                    + s22m1 * y * z * 3. * sqrt((1. - y * y) * (1. - z * z)) * sin_phi / 2.
                    + s220  * (-1. + 3. * y * y) * (-1. + 3. * z * z) / 4.
                    + s221  * y * z * 3. * sqrt((1. - y * y) * (1. - z * z)) * cos_phi / 2.
                    + s222  * y * 3. * (-1. + y * y) * (-1. + z * z) * cos_2phi / 8.
                    );
        }

        double branching_ratio(const double & q2, const double & k2) const
        {
            return decay_width(q2, k2) * tau_B / hbar;
        }

        double branching_ratio_5diff(const double & q2, const double & k2, const double & y/*cos(theta_W)*/, const double & z/*cos(theta_V)*/, const double & phi) const
        {
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
