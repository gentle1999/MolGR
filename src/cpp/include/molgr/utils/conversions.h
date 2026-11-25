#pragma once

#include "molgr/utils/utils.h"

#include <memory>

namespace molgr
{
    namespace utils
    {
        /// Copy atom and bond topology without sharing mutable Open Babel objects.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Independent copy of the molecular topology and explicit electron labels.
        OpenBabel::OBMol CloneMolTopologyOnly(const OpenBabel::OBMol &mol);
        /// Make a topology copy suitable for force-field scoring without changing the input graph.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Independent copy of the molecular topology and explicit electron labels.
        OpenBabel::OBMol CloneMolTopologyForScoring(const OpenBabel::OBMol &mol);
        /// Copy atom and bond topology without sharing mutable Open Babel objects.
        /// @param mol Molecular graph supplying the coordinates, connectivity and atom labels.
        /// @return Independent copy of the molecular topology and explicit electron labels.
        std::shared_ptr<OpenBabel::OBMol> CloneMolTopologyShared(const OpenBabel::OBMol &mol);
        /// Build the covalent Open Babel projection from explicit atom records.
        /// @param data Molecular records with explicit atoms, including hydrogens.
        /// @return Covalent projection with coordination edges retained only in the source records.
        OpenBabel::OBMol MolFromMoleculeData(const MoleculeData &data);
        /// Copy atoms, bonds, explicit electron labels and supported stereo from an OBMol.
        /// @param mol Source graph in Open Babel atom order.
        /// @param assume_electron_properties_known Whether to mark all three electron labels as known.
        /// @return Transferable molecular records with one-based atom indices.
        MoleculeData MoleculeDataFromOBMol(
            const OpenBabel::OBMol &mol,
            bool assume_electron_properties_known);
    }
}
