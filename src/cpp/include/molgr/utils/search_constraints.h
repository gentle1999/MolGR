#pragma once

#include "molgr/compat/openbabel_iter.h"
#include "molgr/utils/electrons.h"
#include <openbabel/bond.h>
#include <openbabel/mol.h>

namespace molgr::search_constraints
{
    inline constexpr const char *kRequiredNitrosoniumOxygen = "MOLGR_REQUIRED_NITROSONIUM_OXYGEN";

    /// Carry an approved NO+ hypothesis through copies of a ligand seed.
    /// @param nitrogen Atom or site occupying this position in the local chemical motif.
    /// @param oxygen Atom or site occupying this position in the local chemical motif.
    inline void RequireNitrosonium(OpenBabel::OBAtom &nitrogen, OpenBabel::OBAtom &oxygen)
    {
        molgr::utils::SetAtomIntegerData(nitrogen, kRequiredNitrosoniumOxygen, oxygen.GetIdx());
    }

    /// Molecule-local ligand hypothesis; intermediate states remain traversable.
    /// @param mol Molecular graph on which the operation acts.
    /// @return True when every recorded ligand-site constraint is satisfied.
    inline bool Satisfied(OpenBabel::OBMol &mol)
    {
        FOR_ATOMS_OF_MOL(n, mol)
        {
            auto *nitrogen = &*n;
            const int index = molgr::utils::GetAtomIntegerData(*nitrogen, kRequiredNitrosoniumOxygen);
            if (index == 0) continue;
            if (index < 1 || index > static_cast<int>(mol.NumAtoms())) return false;
            auto *oxygen = mol.GetAtom(index);
            auto *bond = nitrogen->GetBond(oxygen);
            if (nitrogen->GetAtomicNum() != 7 || oxygen->GetAtomicNum() != 8 ||
                nitrogen->GetExplicitDegree() != 1 || oxygen->GetExplicitDegree() != 1 ||
                nitrogen->GetFormalCharge() != 0 || oxygen->GetFormalCharge() != 1 ||
                !bond || bond->GetBondOrder() != 3) return false;
            for (auto *atom : {nitrogen, oxygen})
                if (atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return false;
        }
        return true;
    }
}
