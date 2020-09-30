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

namespace eos
{
    KKvD2021FormFactors::KKvD2021FormFactors(const Parameters &, const Options &)
    {
    }

    FormFactors<PToGammaOffShell> *
    KKvD2021FormFactors::make(const Parameters & p, const Options & o)
    {
        return new KKvD2021FormFactors(p, o);
    }

    complex<double>
    KKvD2021FormFactors::F_perp(const double & q2, const double & k2) const
    {
        return 1.0; //TODO(SK) -> implement F_perp parametrization
    }

    complex<double>
    KKvD2021FormFactors::F_para(const double & q2, const double & k2) const
    {
        return 1.0; //TODO(SK) -> implement F_para parametrization
    }

    complex<double>
    KKvD2021FormFactors::F_long(const double & q2, const double & k2) const
    {
        return 1.0; //TODO(SK) -> implement F_long parametrization
    }
}
