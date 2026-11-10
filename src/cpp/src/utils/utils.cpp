/**
 * @file utils.cpp
 * @brief Implementation of geometric utility functions and molecule-data helpers.
 * @namespace molgr::utils
 * @author TMJ
 * @date 2025-12-27
 */

#include "molgr/utils/utils.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/electrons.h"

#include <openbabel/mol.h>
#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include "molgr/compat/openbabel_iter.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace molgr
{
    namespace utils
    {

        inline double LengthSq(const OpenBabel::vector3 &v)
        {
            return OpenBabel::dot(v, v);
        }

        double CalculateTetrahedronVolume(const OpenBabel::vector3 &p1, const OpenBabel::vector3 &p2,
                                          const OpenBabel::vector3 &p3, const OpenBabel::vector3 &p4)
        {
            OpenBabel::vector3 v1 = p1 - p4;
            OpenBabel::vector3 v2 = p2 - p4;
            OpenBabel::vector3 v3 = p3 - p4;
            OpenBabel::vector3 cross_product = OpenBabel::cross(v2, v3);
            double scalar_triple_product = OpenBabel::dot(v1, cross_product);
            return std::abs(scalar_triple_product) / 6.0;
        }

        double CalculateShapeQuality(const OpenBabel::vector3 &p1, const OpenBabel::vector3 &p2,
                                     const OpenBabel::vector3 &p3, const OpenBabel::vector3 &p4)
        {
            double volume = CalculateTetrahedronVolume(p1, p2, p3, p4);
            if (volume < 1e-9)
                return 0.0;

            double edges_sq_sum = 0.0;
            edges_sq_sum += LengthSq(p1 - p2);
            edges_sq_sum += LengthSq(p1 - p3);
            edges_sq_sum += LengthSq(p1 - p4);
            edges_sq_sum += LengthSq(p2 - p3);
            edges_sq_sum += LengthSq(p2 - p4);
            edges_sq_sum += LengthSq(p3 - p4);

            double l_rms_squared = edges_sq_sum / 6.0;
            double l_rms_cubed = std::pow(l_rms_squared, 1.5);

            const double NORMALIZATION_CONST = 6.0 * std::sqrt(2.0);
            if (l_rms_cubed < 1e-9)
                return 0.0;
            double quality = NORMALIZATION_CONST * (volume / l_rms_cubed);
            return std::max(0.0, std::min(1.0, quality));
        }

        OpenBabel::vector3 ToVector3(const std::vector<double> &coords)
        {
            if (coords.size() < 3)
                return OpenBabel::vector3(0, 0, 0);
            return OpenBabel::vector3(coords[0], coords[1], coords[2]);
        }

        MoleculeData ExtractMoleculeData(intptr_t mol_ptr)
        {
            OpenBabel::OBMol *mol = reinterpret_cast<OpenBabel::OBMol *>(mol_ptr);
            MoleculeData data;
            data.total_charge = 0;
            data.total_radical_num = 0;

            if (!mol)
            {
                return data;
            }
            return MoleculeDataFromOBMol(*mol);
        }

    } // namespace utils
} // namespace molgr
