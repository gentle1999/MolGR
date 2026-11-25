#pragma once

#include <openbabel/mol.h>

namespace molgr
{
    namespace reconstruct
    {
        /// Move an unresolved two-electron center through an adjacent pi bond.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanCarbeneNeighborUnsaturated(OpenBabel::OBMol &mol);
        /// Resolve adjacent radical-compatible electron states.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param given_charge Signed charge still to assign to the ligand.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanNeighborRadicals(
            OpenBabel::OBMol &mol,
            int given_charge,
            int total_radical_electrons);
        /// Resolve separated terminal radicals or unresolved electron centers.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param given_charge Signed charge still to assign to the ligand.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @return True if at least one bond or electronic assignment changed.
        bool Clean14Radicals(
            OpenBabel::OBMol &mol,
            int given_charge,
            int total_radical_electrons);
        /// Resolve terminal electron states across an alternating six-atom path.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param given_charge Signed charge still to assign to the ligand.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @return True if at least one bond or electronic assignment changed.
        bool Clean16Radicals(
            OpenBabel::OBMol &mol,
            int given_charge,
            int total_radical_electrons);
        /// Convert excess terminal radicals into a neutral 1,3-dipole.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @param given_charge Signed charge still to assign to the ligand.
        /// @param total_radical_electrons Target number of unpaired electrons.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanPossible13Dipole(
            OpenBabel::OBMol &mol,
            int given_charge,
            int total_radical_electrons);
        /// Neutralize the rule-14 triple-bond ion pair and rebuild both endpoints.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanResonances14(OpenBabel::OBMol &mol);
        /// Neutralize the rule-16 conjugated ion pair and rebuild both endpoints.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanResonances16(OpenBabel::OBMol &mol);
        /// Replace a ring allene with its charge-shifted conjugated resonance form.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanResonances17(OpenBabel::OBMol &mol);
        /// Convert an unresolved terminal diazene fragment to a charge-separated azide.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanResonances18(OpenBabel::OBMol &mol);
        /// Apply the ordered local resonance normalizations and report whether the ligand graph changed.
        /// @param mol Ligand graph updated in place when the motif matches.
        /// @return True if at least one bond or electronic assignment changed.
        bool CleanResonances(OpenBabel::OBMol &mol);
    }
}
