#pragma once

#include <openbabel/mol.h>
#include <vector>

namespace molgr
{
    namespace reconstruct
    {
        /// Close N-N-N motifs using explicitly classified endpoint electrons.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @param positive Whether the assignment uses the positive-charge branch.
        /// @return True if at least one bond or electronic assignment changed.
        bool EliminateNNN(OpenBabel::OBMol &mol, int &current_charge_deficit, bool positive = false);
        /// Neutralize positive centers from their most electronegative neighbors.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @param metal_distances Nearest eligible metal-contact distances indexed by ligand atom, in
        ///     milliangstroms.
        /// @return True if at least one bond or electronic assignment changed.
        bool EliminateHighPositiveChargeAtoms(OpenBabel::OBMol &mol, int &current_charge_deficit,
            const std::vector<int> &metal_distances = {});
        /// Convert a carboxyl-like oxygen monoradical into an ``O-`` center.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @param metal_distances Nearest eligible metal-contact distances indexed by ligand atom, in
        ///     milliangstroms.
        /// @return True if at least one bond or electronic assignment changed.
        bool EliminateCarboxyl(OpenBabel::OBMol &mol, int &current_charge_deficit,
            const std::vector<int> &metal_distances = {});
        /// Close an unresolved two-electron center with a heteroatom donor.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @return True if at least one bond or electronic assignment changed.
        bool EliminateCarbeneNeighborHeteroatom(OpenBabel::OBMol &mol, int &current_charge_deficit);
        /// Use positive-charge budget to close a bond in a 1,3-dipole motif.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @return True if at least one bond or electronic assignment changed.
        bool Eliminate13DipolePostive(OpenBabel::OBMol &mol, int &current_charge_deficit);
        /// Localize an excess cyclopentadienyl-like radical as an aromatic anion.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @param metal_distances Nearest eligible metal-contact distances indexed by ligand atom, in
        ///     milliangstroms.
        /// @return True if at least one bond or electronic assignment changed.
        bool EliminatePossibleCPLikeRadicalAnion(
            OpenBabel::OBMol &mol,
            int &current_charge_deficit,
            int total_radical_electrons,
            const std::vector<int> &metal_distances = {});
        /// Consume classified electron sources into positive formal charge.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @return True if at least one bond or electronic assignment changed.
        bool EliminatePositiveCharges(OpenBabel::OBMol &mol, int &current_charge_deficit);
        /// Spend the remaining positive-charge budget while respecting the target unpaired-electron count.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @return True if at least one bond or electronic assignment changed.
        bool EliminatePositiveChargesWithTarget(
            OpenBabel::OBMol &mol,
            int &current_charge_deficit,
            int total_radical_electrons);
        /// Consume classified electron sources into negative formal charge.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param current_charge_deficit Remaining signed charge budget, updated as the stage assigns
        ///     formal charge.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @param metal_distances Nearest eligible metal-contact distances indexed by ligand atom, in
        ///     milliangstroms.
        /// @return True if at least one bond or electronic assignment changed.
        bool EliminateNegativeCharges(
            OpenBabel::OBMol &mol,
            int &current_charge_deficit,
            int total_radical_electrons = -1,
            const std::vector<int> &metal_distances = {});
    }
}
