#pragma once

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/mol.h>
#include <openbabel/ring.h>

namespace molgr
{
    namespace vendor
    {
        namespace openbabel_threading
        {
            /// Infer covalent connectivity from coordinates and then perceive the resulting bond orders in
            /// place.
            /// @param mol Molecule whose native perception data is updated in place.
            void ConnectTheDotsAndPerceiveBondOrders(OpenBabel::OBMol &mol);
            /// Create plausible seed bonds from interatomic distances and prune overcoordinated atoms.
            /// @param mol Molecule whose native perception data is updated in place.
            void ConnectTheDotsLocal(OpenBabel::OBMol &mol);
            /// Perceive bond multiplicities on the existing connectivity using geometry and functional
            /// groups.
            /// @param mol Molecule whose native perception data is updated in place.
            void PerceiveBondOrdersLocal(OpenBabel::OBMol &mol);
            /// Identify ring membership on the current molecular topology.
            /// @param mol Molecule whose native perception data is updated in place.
            void FindRingAtomsAndBonds(OpenBabel::OBMol &mol);
            /// Prepare the ring and aromaticity information required by SMARTS predicates.
            /// @param mol Molecule whose native perception data is updated in place.
            void PrepareForSmartsMatching(OpenBabel::OBMol &mol);
            /// Populate atom hybridization before geometry-dependent queries.
            /// @param mol Molecule whose native perception data is updated in place.
            void EnsureHybridizationPerceived(OpenBabel::OBMol &mol);
            /// Set the molecule's aromaticity-perceived flag.
            /// @param mol Molecule whose native perception data is updated in place.
            /// @param perceived Value to assign to the aromaticity-perceived flag.
            void SetAromaticPerceived(OpenBabel::OBMol &mol, bool perceived);
            /// Recompute aromatic atom and bond flags after topology or electronic labels change.
            /// @param mol Molecule whose native perception data is updated in place.
            void ResetAndAssignAromaticFlags(OpenBabel::OBMol &mol);
            /// @param atom Atom whose local chemical environment is examined.
            /// @return True if the atom is flagged aromatic in its perceived molecule.
            bool AtomIsAromatic(OpenBabel::OBAtom &atom);
            /// @param bond Bond whose endpoints and order define the local environment.
            /// @return True if this bond is flagged aromatic in its perceived molecule.
            bool BondIsAromatic(OpenBabel::OBBond &bond);
            /// @param ring Ring path in the current molecule.
            /// @return True if every atom on the ring path exists and is aromatic.
            bool RingIsAromatic(OpenBabel::OBRing &ring);
        }
    }
}
