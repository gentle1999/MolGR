#pragma once

#include <openbabel/mol.h>

namespace molgr
{
    namespace reconstruct
    {
        /// Remove overlong seed bonds and reconnect partners using their updated valences.
        /// @param mol Molecular graph on which the operation acts.
        /// @param extra_tolerance_angstrom Additional distance allowed beyond the covalent cutoff, in
        ///     angstroms.
        /// @return True if at least one bond or electronic assignment changed.
        bool MakeConnections(OpenBabel::OBMol &mol, double extra_tolerance_angstrom = 0.15);
        /// Apply cheap structural fixes that simplify the later heuristic stages.
        /// @param mol Molecular graph on which the operation acts.
        /// @return True if at least one bond or electronic assignment changed.
        bool PreClean(OpenBabel::OBMol &mol);
        /// Resolve deferred two-electron centers against the global radical budget.
        /// @param mol Molecular graph on which the operation acts.
        /// @param total_charge Target net formal charge of the full input.
        /// @param total_radical Target number of unpaired electrons.
        /// @param emit_warnings Whether to emit diagnostics for rejected electronic assignments.
        /// @return True when the final ligand graph satisfies the electronic targets and structural checks.
        bool ValidateOmol(
            OpenBabel::OBMol &mol,
            int total_charge,
            int total_radical,
            bool emit_warnings = false);
    }
}
