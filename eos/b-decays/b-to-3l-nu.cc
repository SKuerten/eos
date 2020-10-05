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

namespace eos
{
    using std::norm;

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
        }

        double kaellen(const double & s1, const double & s2, const double & s3) const
        {
            return s1 * s1 + s2 * s2 + s3 * s3 - 2.0 *(s1 * s2 + s2 * s3 + s1 * s3);
        }

        double decay_width(const double & q2, const double & k2) const
        {
            const WilsonCoefficients<ChargedCurrent> wc = model->wilson_coefficients_b_to_u(opt_l2.value(), false);

            const complex<double> cVL = wc.cvl();
            const complex<double> cVR = wc.cvr();

            // TODO(SK) -> implement the q^2 and k^2 differential branching
            // fraction
            return power_of<2>(g_fermi * std::abs(model->ckm_ub()) * std::abs(cVL)) * alpha_qed *  (32.0 * M_PI)
                    * std::sqrt(kaellen(q2, 0.0, m_l2 * m_l2)) * std::sqrt(kaellen(k2, m_l1 * m_l1, m_l1 * m_l1)) * std::sqrt(kaellen(m_B * m_B, q2, k2))
                    * (std::norm(form_factors->F_perp(q2, k2)) + std::norm(form_factors->F_para(q2, k2)) + std::norm(form_factors->F_long(q2, k2)));
        }

        // TODO(SK) -> implement the 5 (q^2,k^2,cos(theta_W), cos(theta_V) and phi) differential branching
        double decay_width_5diff(const double & q2, const double & k2, const double & y/*cos(theta_W)*/, const double & z/*cos(theta_V)*/, const double & phi) const
        {
            const WilsonCoefficients<ChargedCurrent> wc = model->wilson_coefficients_b_to_u(opt_l2.value(), false);

            const complex<double> cVL = wc.cvl();
            const complex<double> cVR = wc.cvr();

            return power_of<2>(g_fermi * std::abs(model->ckm_ub()) * std::abs(cVL)) * alpha_qed *  (32.0 * M_PI)
                    * std::sqrt(kaellen(q2, 0.0, m_l2 * m_l2)) * std::sqrt(kaellen(k2, m_l1 * m_l1, m_l1 * m_l1)) * std::sqrt(kaellen(m_B * m_B, q2, k2)) * (
                    (1.0 - z) * cos(2.0 * phi) * (9.0 * std::norm(form_factors->F_para(q2, k2)) - 3.0 * std::norm(form_factors->F_perp(q2, k2))) / (64.0 * M_PI)
                    + 9.0 * (1.0 - z) * (2.0 * std::norm(form_factors->F_long(q2, k2)) - std::norm(form_factors->F_para(q2, k2)) + std::norm(form_factors->F_perp(q2, k2))) / (64.0 *M_PI)
                    + z*z * (9.0 * (y*y * std::norm(form_factors->F_long(q2, k2)) + std::norm(form_factors->F_para(q2, k2))) + std::norm(form_factors->F_perp(q2, k2))) / (32.0 *  M_PI)
                    + y * 3.0 * std::sqrt(3.0) * ((1.0 - z) / 16.0 - 1.0 / 8.0) * std::real(std::conj(form_factors->F_perp(q2, k2)) * form_factors->F_para(q2, k2)) / (M_PI)
                    - 3.0 * std::sqrt(3.0) * ((1.0 - z) * cos(phi) * sin(phi) * std::imag(std::conj(form_factors->F_perp(q2, k2)) * form_factors->F_para(q2, k2)) / (16.0 * M_PI))
                    + y*y * (9.0 * (1.0 - z) * (std::norm(form_factors->F_para(q2, k2)) - std::norm(form_factors->F_perp(q2, k2))) / (64.0 * M_PI)
                    + (1.0 - z) * cos(2.0 * phi)* (3.0 * std::norm(form_factors->F_perp(q2, k2)) - 9.0 * std::norm(form_factors->F_para(q2, k2))) / (64.0 * M_PI)
                    + (3.0 * std::norm(form_factors->F_perp(q2, k2)) + 9.0 * std::norm(form_factors->F_para(q2, k2)) - 9.0 *std::norm(form_factors->F_long(q2, k2))) / (32.0 * M_PI)
                    + 3.0 * std::sqrt(3.0) * (1.0 - z) * cos(phi) * sin(phi) * std::imag(std::conj(form_factors->F_perp(q2, k2)) * form_factors->F_para(q2, k2)) / (16.0 * M_PI))
                    + z * (3.0 * std::sqrt(3.0 * (1.0 - y) * (1.0 - z)) * cos(phi) * std::real(std::conj(form_factors->F_perp(q2, k2)) * form_factors->F_long(q2, k2)) / (16.0 * M_PI)
                    - 9.0 * std::sqrt((1.0 - y) * (1.0 - z)) * sin(phi) * std::imag(std::conj(form_factors->F_para(q2, k2)) * form_factors->F_long(q2, k2)) / (16.0 * M_PI)
                    + y * std::sqrt((1.0 - y) * (1.0 - z)) * (-9.0 * cos(phi) * std::real(std::conj(form_factors->F_para(q2, k2)) * form_factors->F_long(q2, k2))
                    + 3.0 * std::sqrt(3.0) * sin(phi) * std::imag(std::conj(form_factors->F_perp(q2, k2)) * form_factors->F_long(q2, k2))) / (16.0 * M_PI))
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
