#pragma once

#include "molgr/utils/consts.h"
#include "molgr/utils/smarts.h"

#include <openbabel/mol.h>

#include <vector>

namespace molgr
{
    namespace reconstruct
    {
        /// Look up the reconstruction valence and outer-shell information for an element.
        /// @param atomic_num Atomic number of the element.
        /// @return Borrowed element information, or null for an unlisted atomic number.
        const ElementInfo *GetElementInfo(int atomic_num);
        /// Flatten SMARTS match atom indices for subsequent local bond or electron edits.
        /// @param mol Molecular graph on which the operation acts.
        /// @param pattern_id Identifier of the compiled reconstruction SMARTS.
        /// @return One-based atom indices from the matched substructures.
        std::vector<int> GetFlatAtomList(OpenBabel::OBMol &mol, molgr::smarts::PatternId pattern_id);
        /// @param atom_indices Atom indices defining the selected part of the molecular graph.
        /// @param atom_idx One-based Open Babel atom index.
        /// @return Whether the specified atom index occurs in the supplied match.
        bool ContainsAtomIdx(const std::vector<int> &atom_indices, int atom_idx);
    }
}
