/* vim: set sw=4 sts=4 et foldmethod=syntax : */

/*
 * Copyright (c) 2016 Danny van Dyk
 * Copyright (c) 2018 Danny van Dyk
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

#include <test/test.hh>
#include <eos/form-factors/parametric-kkvd2021.hh>

#include <cmath>
#include <limits>
#include <vector>

using namespace test;
using namespace eos;

class ParametricKKvD2021FormFactorTest :
    public TestCase
{
    public:
        ParametricKKvD2021FormFactorTest() :
            TestCase("parametric_kkvd2021_form_factor_test")
        {
        }

        virtual void run() const
        {
            static const double eps = 1.0e-5;

            Parameters p = Parameters::Defaults();
            p["B->gamma^*::t_0@KKvD2021"] = +3.1097e+01;
            p["B->gamma^*::N^1_perp_0@KKvD2021"] = +1.0000e+00;
            p["B->gamma^*::N^1_perp_1@KKvD2021"] = +1.0000e+00;
            p["B->gamma^*::N^1_perp_2@KKvD2021"] = +1.0000e+00;

            {
                KKvD2021FormFactors kkvd2021(p, Options{ });

                TEST_CHECK_NEARLY_EQUAL( 1.09947, kkvd2021.F_perp(1.0, 0.05),  eps);
            }
        }
} parametric_kkvd2021_form_factor_test;
