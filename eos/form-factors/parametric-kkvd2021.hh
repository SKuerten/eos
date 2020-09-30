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

#ifndef EOS_GUARD_SRC_FORM_FACTORS_PARAMETRIC_KKVD2021_HH
#define EOS_GUARD_SRC_FORM_FACTORS_PARAMETRIC_KKVD2021_HH 1

#include <eos/form-factors/mesonic.hh>
#include <eos/utils/complex.hh>
#include <eos/utils/parameters.hh>
#include <eos/utils/options.hh>
#include <eos/utils/qualified-name.hh>

#include <memory>

namespace eos
{
    class KKvD2021FormFactors :
        public FormFactors<PToGammaOffShell>
    {
        private:

        public:
            KKvD2021FormFactors(const Parameters &, const Options &);

            static FormFactors<PToGammaOffShell> * make(const Parameters &, const Options &);

            virtual complex<double> F_perp(const double & q2, const double & k2) const;
            virtual complex<double> F_para(const double & q2, const double & k2) const;
            virtual complex<double> F_long(const double & q2, const double & k2) const;
    };
}

#endif
