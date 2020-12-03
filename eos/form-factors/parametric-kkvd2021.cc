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

#include <eos/form-factors/parametric-kkvd2021.hh>
#include <eos/utils/power_of.hh>
#include <eos/utils/integrate.hh>

namespace eos
{
    KKvD2021FormFactors::KKvD2021FormFactors(const Parameters & p, const Options &) :
        _t_0(p["B->gamma^*::t_0@KKvD2021"], *this),
        _N_1_perp_0(p["B->gamma^*::N^1_perp_0@KKvD2021"], *this),
        _N_1_perp_1(p["B->gamma^*::N^1_perp_1@KKvD2021"], *this),
        _N_1_perp_2(p["B->gamma^*::N^1_perp_2@KKvD2021"], *this)
    {
    }

    FormFactors<PToGammaOffShell> *
    KKvD2021FormFactors::make(const Parameters & p, const Options & o)
    {
        return new KKvD2021FormFactors(p, o);
    }

    double
    KKvD2021FormFactors::_k(const double & s) const
    {
        const double mPion = 0.13957;
 
        return sqrt(s/4.0 - mPion * mPion);
    }

    double
    KKvD2021FormFactors::_wP(const double & s) const
    {
        const double s0P = 1.05 * 1.05;
 
        return (sqrt(s) - sqrt(s0P - s)) / (sqrt(s) + sqrt(s0P -s));
    }
 
    double
    KKvD2021FormFactors::_lowenergy1(const double & s) const
    {
        const double mPion = 0.13957;
        const double mRho  = 0.7736;
        const double b0P   = 1.043;
        const double b1P   = 0.19;
 
        return atan(1./(sqrt(s) / (2.0 * power_of<3>(_k(s) ) ) * (mRho * mRho - s)
            * (2.0 * power_of<3>(mPion) / (sqrt(s) * mRho * mRho) + b0P + b1P * _wP(s) ) ) );
    }

    double
    KKvD2021FormFactors::_lowenergy2(const double & s) const
    {
        const double mPion = 0.13957;
        const double mRho  = 0.7736;
        const double b0P   = 1.043;
        const double b1P   = 0.19;
 
        return atan(1./(sqrt(s) / (2.0 * power_of<3>(_k(s) ) ) * (mRho * mRho - s)
            * (2.0 * power_of<3>(mPion) / (sqrt(s) * mRho * mRho) + b0P + b1P * _wP(s) ) ) ) + M_PI;
    }
 
    double
    KKvD2021FormFactors::_highenergy(const double & s) const
    {
        const double mKaon    =  0.496;
        const double lambda0P =  2.6838;
        const double lambda1P =  1.39;
        const double lambda2P = -1.7;

        const double breakup_factor = (sqrt(s) / (2.0 * mKaon)  -1.0);

        return lambda0P + lambda1P * breakup_factor + lambda2P * breakup_factor * breakup_factor;
    }

    double
    KKvD2021FormFactors::_continuationP(const double & s) const
    {
        return M_PI - 1.26587 / (5.36287 + s);
    }

    double
    KKvD2021FormFactors::_phaseshiftP(const double & s) const
    {
        if (s < 0.598457) return _lowenergy1(s);
        else if (0.598457 <= s && s < 0.984064) return _lowenergy2(s);
        else if (0.984064 <= s && s < 2.0164) return _highenergy(s);
        else if ( 2.0164 <= s) return _continuationP(s);
        else return 1; //Add Error handling
    }

    double
    KKvD2021FormFactors::_omega_integrand(const double & t, const double & s) const
    {
        const double mPion = 0.13957;

        double x = 4.0 * mPion * mPion + (1.0 - t) / t;
        double x_jacobian = 1.0 / (t * t);// jacobian is missing a minus sign cause the gsl-integration takes care of it by the integration interval

        return x_jacobian * _phaseshiftP(x) / (x * (x - s) );
    }

    double
    KKvD2021FormFactors::_omega(const double & k2) const
    {
        std::function<double(const double &)> f = std::bind(&KKvD2021FormFactors::_omega_integrand, this, std::placeholders::_1, k2);

        const double mPion = 0.13957;
        double t_sing = 1.0 / (k2 - 4.0 * mPion * mPion + 1.0);

        if (k2 < 4.0 * mPion * mPion)
        {
            auto config_QNG = GSL::QNG::Config().epsabs(1.0e-7);

            return exp((k2 / M_PI) * (integrate<GSL::QNG>(f, 0.0, 1.0, config_QNG)));
        }

        if (k2 >= 4.0 * mPion * mPion)
        {
            auto config_QAGS = GSL::QAGS::Config().epsrel(1.0e-7);

            return exp((k2 / M_PI) * (integrate<GSL::QAGS>(f, 1.0e-7, t_sing - 1.0e-7, config_QAGS) + integrate<GSL::QAGS>(f, t_sing + 1.0e-7, 1.0, config_QAGS) ) );
        }

        else
        {
        //throw error or something
        }
    }

    double
    KKvD2021FormFactors::_z(const double & q2) const
    {
        static const double t_p = power_of<2>(5.279 + 2.0 * 0.137);
        const double t_0 = t_p * ( 1.0 - sqrt(1 - q2 / t_p) );//Value chosen such that abs(z(q2)-z(0)) is minimal over the q2 range

        const double tp = sqrt(t_p -q2);
        const double t0 = sqrt(t_0 -q2);

        return (tp - t0) / (tp + t0);
    }

    complex<double>
    KKvD2021FormFactors::F_perp(const double & q2, const double & k2) const
    {
        return _omega(k2)*q2; //TODO(SK) -> implement F_perp parametrization
    }

    complex<double>
    KKvD2021FormFactors::F_para(const double & q2, const double & k2) const
    {
        return 1.0*q2*k2; //TODO(SK) -> implement F_para parametrization
    }

    complex<double>
    KKvD2021FormFactors::F_long(const double & q2, const double & k2) const
    {
        return 1.0*q2*k2; //TODO(SK) -> implement F_long parametrization
    }
}
