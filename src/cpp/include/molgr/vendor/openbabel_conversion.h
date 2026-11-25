#pragma once

#include <openbabel/mol.h>

#include <string>

namespace molgr
{
    namespace vendor
    {
        namespace openbabel_conversion
        {
            // Open Babel conversion mutates process-global option/plugin registries.
            // Keep the remaining SMILES compatibility surface in one adapter.
            /// Parse SMILES into the supplied molecule while holding the native conversion lock.
            /// @param smiles SMILES representation of the input molecular graph.
            /// @param mol Destination molecule receiving the parsed atoms and bonds.
            /// @return True when the entire requested input was successfully read.
            bool ReadSmiles(const std::string &smiles, OpenBabel::OBMol *mol);
            /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
            /// @return The first SMILES token produced for the molecule, excluding any trailing title.
            std::string WriteSmilesFirstToken(const OpenBabel::OBMol &mol);
        }
    }
}
