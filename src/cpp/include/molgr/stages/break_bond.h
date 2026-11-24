#pragma once

#include <openbabel/mol.h>

namespace molgr
{
    namespace reconstruct
    {
        /// Homolytically reduce deformed multiple bonds by one bond-order unit.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param allowed_charge Charge budget reserved during bond breaking.
        /// @param allowed_radical Target unpaired-electron budget used to decide whether bond breaking is
        ///     needed.
        /// @param tolerance Minimum torsional deviation in degrees for a deformed pi bond.
        /// @return True if at least one bond or electronic assignment changed.
        bool BreakDeformedEne(
            OpenBabel::OBMol &mol,
            int allowed_charge,
            int allowed_radical,
            double tolerance = 5.0);
        /// Apply last-resort homolytic and charge-transfer bond-breaking templates.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @param allowed_radical Target unpaired-electron budget used to decide whether bond breaking is
        ///     needed.
        /// @return True if at least one bond or electronic assignment changed.
        bool BreakOneBond(
            OpenBabel::OBMol &mol,
            int &current_charge_deficit,
            int allowed_radical);
    }
}
