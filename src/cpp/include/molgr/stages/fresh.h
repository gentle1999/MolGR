#pragma once

#include <openbabel/atom.h>
#include <openbabel/mol.h>

namespace molgr
{
    namespace reconstruct
    {
        /// @param atom Atom whose local chemical environment is examined.
        /// @return The local bond-valence deficit, before classifying its electrons.
        int AssignRadicalDots(OpenBabel::OBAtom &atom);
        /// Rebuild one atom's charge and explicit electron classification.
        /// @param atom Atom whose local chemical environment is examined.
        /// @return True if the atom's formal charge or explicit electron classification changed.
        bool AssignChargeRadicalForAtom(OpenBabel::OBAtom &atom);
        /// Apply local charge/electron rebuilding independently to every atom.
        /// @param mol Molecular graph on which the operation acts.
        /// @return True if at least one bond or electronic assignment changed.
        bool FreshOmolChargeRadical(OpenBabel::OBMol &mol);
    }
}
