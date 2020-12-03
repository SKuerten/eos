/* vim: set sw=4 sts=4 et foldmethod=syntax : */

/*
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
#include <eos/b-decays/b-to-3l-nu.hh>

using namespace test;
using namespace eos;

class BToThreeLeptonsNeutrinoTest :
    public TestCase
{
    public:
        BToThreeLeptonsNeutrinoTest() :
            TestCase("b_to_3l_nu_test")
        {
        }

        virtual void run() const
        {

            static const double eps = 1e-5;

            Parameters p = Parameters::Defaults();

            p["B(*)->D(*)::xi'(1)@HQET"].set(-1.06919);
            p["B(*)->D(*)::xi''(1)@HQET"].set(1.66581);
            p["B(*)->D(*)::xi'''(1)@HQET"].set(-2.91356);

            {

                BToThreeLeptonsNeutrino three_lnu(p, Options{ });


                TEST_CHECK_NEARLY_EQUAL(three_lnu.branching_ratio_5diff(2.0, 2.0, 1.0, 1.0, 0.1), 0.669598,     eps);
            }
        }
} b_to_3l_nu_test;
