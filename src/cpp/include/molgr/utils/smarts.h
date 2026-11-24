#pragma once

#include <openbabel/mol.h>
#include <openbabel/parsmart.h>

#include <cstddef>
#include <vector>

namespace molgr
{
    namespace smarts
    {
        /// Identifier of a compiled substructure pattern shared by native reconstruction stages.
        enum class PatternId : std::size_t
        {
            CONNECTION_GENERAL_DONOR = 0,
            CONNECTION_GENERAL_PARTNER,
            CONNECTION_P_V2,
            CONNECTION_P_PARTNER,
            CONNECTION_O_V1,
            CONNECTION_C_V3,
            CONNECTION_SE_I_DONOR,
            CONNECTION_O_PARTNER,
            CONNECTION_P_FIVE_COORDINATE,
            CONNECTION_P_SIXTH_PARTNER,
            LONG_SI_CARBONYL_BOND,
            LONG_TETRACOORDINATE_P_P_BOND,
            PRE_CLEAN_HYPERVALENT,
            PRE_CLEAN_HYPER_PI_BOND,
            PRE_CLEAN_BCP_RING_5,
            PRE_CLEAN_BCP_RING_4,
            ELIM_HIGH_POSITIVE,
            ELIM_CARBOXYL,
            ELIM_NNN_NEGATIVE,
            ELIM_NNN_POSITIVE,
            ELIM_1_3_DIPOLE_POSTIVE,
            ELIM_POSITIVE_N,
            ELIM_POSITIVE_C_H,
            ELIM_POSITIVE_DIPOLE,
            ELIM_NEGATIVE_F,
            ELIM_NEGATIVE_O,
            ELIM_NEGATIVE_O_1,
            ELIM_NEGATIVE_CL,
            ELIM_NEGATIVE_N,
            ELIM_NEGATIVE_N_1,
            ELIM_NEGATIVE_N_2,
            ELIM_NEGATIVE_BR,
            ELIM_NEGATIVE_I,
            ELIM_NEGATIVE_S,
            ELIM_NEGATIVE_S_1,
            ELIM_NEGATIVE_SE,
            ELIM_NEGATIVE_SE_1,
            ELIM_NEGATIVE_P,
            ELIM_NEGATIVE_P_1,
            ELIM_NEGATIVE_P_2,
            ELIM_NEGATIVE_B,
            ELIM_NEGATIVE_B_1,
            ELIM_NEGATIVE_B_2,
            ELIM_NEGATIVE_C_V3,
            ELIM_NEGATIVE_H,
            ELIM_NEGATIVE_C_LOW,
            ELIM_NEGATIVE_CP,
            CLEAN_CARBENE_NEIGHBOR_UNSAT,
            CLEAN_POSSIBLE_1_3_DIPOLE,
            CLEAN_1_4_RADICALS,
            CLEAN_1_6_RADICALS,
            CLEAN_RESONANCE_0,
            CLEAN_RESONANCE_1,
            CLEAN_RESONANCE_2,
            CLEAN_RESONANCE_3,
            CLEAN_RESONANCE_4,
            CLEAN_RESONANCE_5,
            CLEAN_RESONANCE_6,
            CLEAN_RESONANCE_7,
            CLEAN_RESONANCE_8,
            CLEAN_RESONANCE_9,
            CLEAN_RESONANCE_10,
            CLEAN_RESONANCE_11,
            CLEAN_RESONANCE_12,
            CLEAN_RESONANCE_13,
            CLEAN_RESONANCE_14,
            CLEAN_RESONANCE_16,
            CLEAN_RESONANCE_17,
            CLEAN_RESONANCE_18,
            BREAK_DEFORMED_ENE_A,
            BREAK_DEFORMED_ENE_B,
            BREAK_ONE_BOND_MULTIPLE,
            BREAK_ONE_BOND_CATION,
            BREAK_ONE_BOND_AROMATIC,
            RESONANCE_ONE_STEP,
            SCORING_CONJUGATION,
            COUNT,
        };

        // Match SMARTS with the same semantics as Python pybel.Smarts.findall():
        // OpenBabel::OBSmartsPattern::Match(mol), followed by GetUMapList().
        /// @param pattern Substructure pattern to match.
        /// @param mol Molecular graph on which the operation acts.
        /// @return Matching one-based atom index lists in SMARTS traversal order.
        std::vector<std::vector<int>> FindAll(
            OpenBabel::OBSmartsPattern &pattern,
            OpenBabel::OBMol &mol);
        /// @param mol Molecular graph on which the operation acts.
        /// @param pattern_id Identifier of the compiled reconstruction SMARTS.
        /// @return Matching one-based atom index lists in SMARTS traversal order.
        std::vector<std::vector<int>> FindAll(OpenBabel::OBMol &mol, PatternId pattern_id);
    }
}
