#pragma once

#include <openbabel/mol.h>

#include <string>
#include <vector>

namespace molgr
{
    namespace utils
    {
        /// Read XYZ coordinates and perceive the seed topology in the destination OBMol.
        /// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in
        ///     angstroms.
        /// @param mol Destination molecule receiving the parsed atoms and bonds.
        /// @return True when the entire requested input was successfully read.
        bool ReadXyzBlockToMol(const std::string &xyz_block, OpenBabel::OBMol *mol);

        /// Serialize the molecule without entering Open Babel's format/plugin
        /// registry. This is the production-side counterpart to
        /// ReadXyzBlockToMol and is safe to call from independent workers.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return XYZ text with atom coordinates in molecular order.
        std::string WriteXyzBlock(const OpenBabel::OBMol &mol);

        /// Parse the XYZ atom records without touching Open Babel format
        /// plugins. This is used by batch-side electronic-state validation.
        /// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in
        ///     angstroms.
        /// @param atomic_numbers Output atomic numbers in XYZ atom order.
        /// @return True when the entire requested input was successfully read.
        bool ParseXyzAtomicNumbers(
            const std::string &xyz_block,
            std::vector<int> *atomic_numbers);
    }
}
