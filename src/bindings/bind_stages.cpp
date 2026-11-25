#include "bindings.h"

#include "molgr/stages/clean.h"
#include "molgr/stages/break_bond.h"
#include "molgr/stages/eliminate.h"
#include "molgr/stages/fresh.h"
#include "molgr/stages/preprocess.h"
#include "molgr/utils/logger.h"

#include <cstdint>
#include <stdexcept>

namespace molgr {
namespace bind {

/// Borrow a caller-owned OBMol by address; reject a null pointer before accessing atoms or bonds.
/// @param mol_ptr Address of the caller-owned Open Babel molecule.
/// @return Borrowed molecule pointer. A null address raises an exception.
static OpenBabel::OBMol *require_obmol_ptr(intptr_t mol_ptr)
{
    molgr::WarnUnsafeOpenBabelUse(
        "molgr.dev.stages.*_ptr",
        "OBMol pointers are not thread-safe; keep ownership and all operations on one thread, and free owned pointers exactly once.");
    if (mol_ptr == 0)
    {
        throw std::runtime_error("null OBMol pointer");
    }
    return reinterpret_cast<OpenBabel::OBMol *>(mol_ptr);
}

void bind_stages(py::module_ &m)
{
    /// These dev helpers operate on OBMol pointers owned by Python OpenBabel wrappers.
    /// Keep the GIL held while using those pointers and while constructing py::object results.
    const auto make_connections_ptr = [](intptr_t mol_ptr, double extra_tolerance_angstrom)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::MakeConnections(*mol, extra_tolerance_angstrom);
    };

    /// Python adapter for molgr::reconstruct::PreClean. Apply the stage to the borrowed graph and report
    /// whether it changed.
    const auto pre_clean_ptr = [](intptr_t mol_ptr)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::PreClean(*mol);
    };

    /// Python adapter for molgr::reconstruct::FreshOmolChargeRadical. Apply the stage to the borrowed graph
    /// and report whether it changed.
    const auto fresh_omol_charge_radical_ptr = [](intptr_t mol_ptr)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::FreshOmolChargeRadical(*mol);
    };

    /// Python adapter for molgr::reconstruct::AssignRadicalDots. Return the local active-electron deficit
    /// at the one-based atom index.
    const auto assign_radical_dots_ptr = [](intptr_t mol_ptr, int atom_idx) -> int
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        auto *atom = mol->GetAtom(atom_idx);
        if (!atom)
        {
            throw std::runtime_error("invalid atom index");
        }
        return molgr::reconstruct::AssignRadicalDots(*atom);
    };

    /// Python adapter for molgr::reconstruct::AssignChargeRadicalForAtom. Apply the stage to the borrowed
    /// graph and report whether it changed.
    const auto assign_charge_radical_for_atom_ptr = [](intptr_t mol_ptr, int atom_idx)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        auto *atom = mol->GetAtom(atom_idx);
        if (!atom)
        {
            throw std::runtime_error("invalid atom index");
        }
        return molgr::reconstruct::AssignChargeRadicalForAtom(*atom);
    };

    /// Python adapter for molgr::reconstruct::ValidateOmol. Return whether the borrowed graph satisfies
    /// both electronic targets.
    const auto validate_omol_ptr = [](intptr_t mol_ptr, int total_charge, int total_radical) -> bool
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::ValidateOmol(*mol, total_charge, total_radical);
    };

    /// Python adapter for molgr::reconstruct::Eliminate13DipolePostive. Return the outstanding charge and
    /// whether the graph changed.
    const auto eliminate_1_3_dipole_postive_ptr = [](intptr_t mol_ptr,
                                                       int given_charge) -> py::tuple
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        const bool hit = molgr::reconstruct::Eliminate13DipolePostive(*mol, given_charge);
        return py::make_tuple(given_charge, hit);
    };

    /// Python adapter for molgr::reconstruct::EliminatePositiveCharges. Return the outstanding charge and
    /// whether the graph changed.
    const auto eliminate_positive_charges_ptr = [](intptr_t mol_ptr, int given_charge) -> py::tuple
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        const bool hit = molgr::reconstruct::EliminatePositiveCharges(*mol, given_charge);
        return py::make_tuple(given_charge, hit);
    };

    /// Python adapter for molgr::reconstruct::EliminateNegativeCharges. Return the outstanding charge and
    /// whether the graph changed.
    const auto eliminate_negative_charges_ptr = [](
        intptr_t mol_ptr,
        int given_charge,
        int total_radical_electrons) -> py::tuple
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        const bool hit = molgr::reconstruct::EliminateNegativeCharges(
            *mol,
            given_charge,
            total_radical_electrons);
        return py::make_tuple(given_charge, hit);
    };

    /// Python adapter for molgr::reconstruct::EliminatePossibleCPLikeRadicalAnion. Return the outstanding
    /// charge and whether the graph changed.
    const auto eliminate_possible_cp_like_radical_anion_ptr = [](
                                                                 intptr_t mol_ptr,
                                                                 int given_charge,
                                                                 int total_radical_electrons) -> py::tuple
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        const bool hit = molgr::reconstruct::EliminatePossibleCPLikeRadicalAnion(
            *mol,
            given_charge,
            total_radical_electrons);
        return py::make_tuple(given_charge, hit);
    };

    /// Python adapter for molgr::reconstruct::EliminateNNN. Return the outstanding charge and whether the
    /// graph changed.
    const auto eliminate_nnn_ptr = [](intptr_t mol_ptr, int given_charge, bool positive) -> py::tuple
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        const bool hit = molgr::reconstruct::EliminateNNN(*mol, given_charge, positive);
        return py::make_tuple(given_charge, hit);
    };

    /// Python adapter for molgr::reconstruct::EliminateHighPositiveChargeAtoms. Return the outstanding
    /// charge and whether the graph changed.
    const auto eliminate_high_positive_charge_atoms_ptr = [](intptr_t mol_ptr, int given_charge) -> py::tuple
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        const bool hit = molgr::reconstruct::EliminateHighPositiveChargeAtoms(*mol, given_charge);
        return py::make_tuple(given_charge, hit);
    };

    /// Python adapter for molgr::reconstruct::EliminateCarboxyl. Return the outstanding charge and whether
    /// the graph changed.
    const auto eliminate_carboxyl_ptr = [](intptr_t mol_ptr, int given_charge) -> py::tuple
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        const bool hit = molgr::reconstruct::EliminateCarboxyl(*mol, given_charge);
        return py::make_tuple(given_charge, hit);
    };

    /// Python adapter for molgr::reconstruct::EliminateCarbeneNeighborHeteroatom. Return the outstanding
    /// charge and whether the graph changed.
    const auto eliminate_carbene_neighbor_heteroatom_ptr = [](intptr_t mol_ptr, int given_charge) -> py::tuple
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        const bool hit = molgr::reconstruct::EliminateCarbeneNeighborHeteroatom(*mol, given_charge);
        return py::make_tuple(given_charge, hit);
    };

    /// Python adapter for molgr::reconstruct::CleanResonances. Apply the stage to the borrowed graph and
    /// report whether it changed.
    const auto clean_resonances_ptr = [](intptr_t mol_ptr)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::CleanResonances(*mol);
    };

    /// Python adapter for molgr::reconstruct::CleanResonances14. Apply the stage to the borrowed graph and
    /// report whether it changed.
    const auto clean_resonances_14_ptr = [](intptr_t mol_ptr)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::CleanResonances14(*mol);
    };

    /// Python adapter for molgr::reconstruct::CleanResonances16. Apply the stage to the borrowed graph and
    /// report whether it changed.
    const auto clean_resonances_16_ptr = [](intptr_t mol_ptr)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::CleanResonances16(*mol);
    };

    /// Python adapter for molgr::reconstruct::CleanResonances17. Apply the stage to the borrowed graph and
    /// report whether it changed.
    const auto clean_resonances_17_ptr = [](intptr_t mol_ptr)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::CleanResonances17(*mol);
    };

    /// Python adapter for molgr::reconstruct::CleanResonances18. Apply the stage to the borrowed graph and
    /// report whether it changed.
    const auto clean_resonances_18_ptr = [](intptr_t mol_ptr)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::CleanResonances18(*mol);
    };

    /// Python adapter for molgr::reconstruct::CleanNeighborRadicals. Apply the stage to the borrowed graph
    /// and report whether it changed.
    const auto clean_neighbor_radicals_ptr = [](
                                                  intptr_t mol_ptr,
                                                  int given_charge,
                                                  int total_radical_electrons)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::CleanNeighborRadicals(
            *mol,
            given_charge,
            total_radical_electrons);
    };

    /// Python adapter for molgr::reconstruct::Clean14Radicals. Apply the stage to the borrowed graph and
    /// report whether it changed.
    const auto clean_1_4_radicals_ptr = [](
                                             intptr_t mol_ptr,
                                             int given_charge,
                                             int total_radical_electrons)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::Clean14Radicals(
            *mol,
            given_charge,
            total_radical_electrons);
    };

    /// Python adapter for molgr::reconstruct::Clean16Radicals. Apply the stage to the borrowed graph and
    /// report whether it changed.
    const auto clean_1_6_radicals_ptr = [](
                                             intptr_t mol_ptr,
                                             int given_charge,
                                             int total_radical_electrons)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::Clean16Radicals(
            *mol,
            given_charge,
            total_radical_electrons);
    };

    /// Python adapter for molgr::reconstruct::CleanPossible13Dipole. Apply the stage to the borrowed graph
    /// and report whether it changed.
    const auto clean_possible_1_3_dipole_ptr = [](
                                                  intptr_t mol_ptr,
                                                  int given_charge,
                                                  int total_radical_electrons)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::CleanPossible13Dipole(
            *mol,
            given_charge,
            total_radical_electrons);
    };

    /// Python adapter for molgr::reconstruct::CleanCarbeneNeighborUnsaturated. Apply the stage to the
    /// borrowed graph and report whether it changed.
    const auto clean_carbene_neighbor_unsaturated_ptr = [](intptr_t mol_ptr)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::CleanCarbeneNeighborUnsaturated(*mol);
    };

    /// Python adapter for molgr::reconstruct::BreakDeformedEne. Apply the stage to the borrowed graph and
    /// report whether it changed.
    const auto break_deformed_ene_ptr = [](intptr_t mol_ptr, int given_charge, int given_radical, double tolerance)
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        return molgr::reconstruct::BreakDeformedEne(*mol, given_charge, given_radical, tolerance);
    };

    /// Python adapter for molgr::reconstruct::BreakOneBond. Return the outstanding charge and whether the
    /// graph changed.
    const auto break_one_bond_ptr = [](intptr_t mol_ptr, int given_charge, int given_radical) -> py::tuple
    {
        auto *mol = require_obmol_ptr(mol_ptr);
        const bool hit = molgr::reconstruct::BreakOneBond(*mol, given_charge, given_radical);
        return py::make_tuple(given_charge, hit);
    };

    auto m_preprocess = m.def_submodule("preprocess", "Fallback-aligned preprocess stage helpers");
    auto m_fresh = m.def_submodule("fresh", "Fallback-aligned fresh stage helpers");
    auto m_eliminate = m.def_submodule("eliminate", "Fallback-aligned eliminate stage helpers");
    auto m_clean = m.def_submodule("clean", "Fallback-aligned clean stage helpers");
    auto m_break_bond = m.def_submodule("break_bond", "Fallback-aligned break_bond stage helpers");

    m_preprocess.def(
        "make_connections_ptr",
        make_connections_ptr,
        R"pbdoc(
Remove overlong seed bonds and reconnect partners using their updated valences.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    extra_tolerance_angstrom: additive distance tolerance in Angstrom
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("extra_tolerance_angstrom") = 0.15);

    m_preprocess.def(
        "pre_clean_ptr",
        pre_clean_ptr,
        R"pbdoc(
Normalize labile seed bonds and charge-separated groups before assigning electrons.
)pbdoc",
        py::arg("mol_ptr"));

    m_preprocess.def(
        "validate_omol_ptr",
        validate_omol_ptr,
        R"pbdoc(
Validate conservation of total charge and radical electrons.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    total_charge: expected total formal charge
    total_radical: expected total radical electrons
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("total_charge"),
        py::arg("total_radical"));

    m_fresh.def(
        "fresh_omol_charge_radical_ptr",
        fresh_omol_charge_radical_ptr,
        R"pbdoc(
Apply local charge/electron rebuilding independently to every atom.
)pbdoc",
        py::arg("mol_ptr"));

    m_fresh.def(
        "assign_radical_dots_ptr",
        assign_radical_dots_ptr,
        R"pbdoc(
Assign radical dots for a specific atom on an existing OBMol.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    atom_idx: 1-based atom index
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("atom_idx"));

    m_fresh.def(
        "assign_charge_radical_for_atom_ptr",
        assign_charge_radical_for_atom_ptr,
        R"pbdoc(
Assign charge/radical state for a specific atom on an existing OBMol.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    atom_idx: 1-based atom index
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("atom_idx"));

    m_eliminate.def(
        "eliminate_1_3_dipole_postive_ptr",
        eliminate_1_3_dipole_postive_ptr,
        R"pbdoc(
Use positive-charge budget to close a bond in a 1,3-dipole motif.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: charge deficit to be updated in place and returned
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"));

    m_eliminate.def(
        "eliminate_positive_charges_ptr",
        eliminate_positive_charges_ptr,
        R"pbdoc(
Consume classified electron sources into positive formal charge.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: charge deficit to be updated in place and returned
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"));

    m_eliminate.def(
        "eliminate_negative_charges_ptr",
        eliminate_negative_charges_ptr,
        R"pbdoc(
Consume classified electron sources into negative formal charge.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: charge deficit to be updated in place and returned
    total_radical_electrons: optional target number of radical electrons to preserve;
        use -1 to disable target preservation
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"),
        py::arg("total_radical_electrons") = -1);

    m_eliminate.def(
        "eliminate_possible_cp_like_radical_anion_ptr",
        eliminate_possible_cp_like_radical_anion_ptr,
        R"pbdoc(
Localize an excess cyclopentadienyl-like radical as an aromatic anion.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: charge deficit to be updated in place and returned
    total_radical_electrons: target number of unpaired electrons to preserve
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"),
        py::arg("total_radical_electrons"));

    m_eliminate.def(
        "eliminate_nnn_ptr",
        eliminate_nnn_ptr,
        R"pbdoc(
Close N-N-N motifs using explicitly classified endpoint electrons.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: charge deficit to be updated in place and returned
    positive: whether to run positive-direction elimination
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"),
        py::arg("positive") = false);

    m_eliminate.def(
        "eliminate_high_positive_charge_atoms_ptr",
        eliminate_high_positive_charge_atoms_ptr,
        R"pbdoc(
Neutralize positive centers from their most electronegative neighbors.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: charge deficit to be updated in place and returned
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"));

    m_eliminate.def(
        "eliminate_carboxyl_ptr",
        eliminate_carboxyl_ptr,
        R"pbdoc(
Convert a carboxyl-like oxygen monoradical into an O- center.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: charge deficit to be updated in place and returned
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"));

    m_eliminate.def(
        "eliminate_carbene_neighbor_heteroatom_ptr",
        eliminate_carbene_neighbor_heteroatom_ptr,
        R"pbdoc(
Close an unresolved two-electron center with a heteroatom donor.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: charge deficit to be updated in place and returned
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"));

    m_clean.def(
        "clean_resonances_ptr",
        clean_resonances_ptr,
        R"pbdoc(
Run the ordered resonance normalization rule set after candidate generation.
)pbdoc",
        py::arg("mol_ptr"));

    m_clean.def(
        "clean_resonances_14_ptr",
        clean_resonances_14_ptr,
        R"pbdoc(
Neutralize the rule-14 triple-bond ion pair and rebuild both endpoints.
)pbdoc",
        py::arg("mol_ptr"));

    m_clean.def(
        "clean_resonances_16_ptr",
        clean_resonances_16_ptr,
        R"pbdoc(
Neutralize the rule-16 conjugated ion pair and rebuild both endpoints.
)pbdoc",
        py::arg("mol_ptr"));

    m_clean.def(
        "clean_resonances_17_ptr",
        clean_resonances_17_ptr,
        R"pbdoc(
Replace a ring allene with its charge-shifted conjugated resonance form.
)pbdoc",
        py::arg("mol_ptr"));

    m_clean.def(
        "clean_resonances_18_ptr",
        clean_resonances_18_ptr,
        R"pbdoc(
Convert an unresolved terminal diazene fragment to a charge-separated azide.
)pbdoc",
        py::arg("mol_ptr"));

    m_clean.def(
        "clean_possible_1_3_dipole_ptr",
        clean_possible_1_3_dipole_ptr,
        R"pbdoc(
Convert an eligible excess-radical fragment into a neutral 1,3-dipole.
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"),
        py::arg("total_radical_electrons"));

    m_clean.def(
        "clean_neighbor_radicals_ptr",
        clean_neighbor_radicals_ptr,
        R"pbdoc(
Resolve adjacent radical-compatible electron states.
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"),
        py::arg("total_radical_electrons"));

    m_clean.def(
        "clean_1_4_radicals_ptr",
        clean_1_4_radicals_ptr,
        R"pbdoc(
Resolve separated terminal radicals or unresolved electron centers.
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"),
        py::arg("total_radical_electrons"));

    m_clean.def(
        "clean_1_6_radicals_ptr",
        clean_1_6_radicals_ptr,
        R"pbdoc(
Resolve terminal electron states across an alternating six-atom path.
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge"),
        py::arg("total_radical_electrons"));

    m_clean.def(
        "clean_carbene_neighbor_unsaturated_ptr",
        clean_carbene_neighbor_unsaturated_ptr,
        R"pbdoc(
Move an unresolved two-electron center through an adjacent pi bond.
)pbdoc",
        py::arg("mol_ptr"));

    m_break_bond.def(
        "break_deformed_ene_ptr",
        break_deformed_ene_ptr,
        R"pbdoc(
Homolytically reduce deformed multiple bonds by one bond-order unit.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: target charge budget used by the stage
    given_radical: target radical budget used by the stage
    tolerance: torsion-angle tolerance threshold
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge") = 0,
        py::arg("given_radical") = 0,
        py::arg("tolerance") = 5.0);

    m_break_bond.def(
        "break_one_bond_ptr",
        break_one_bond_ptr,
        R"pbdoc(
Apply last-resort homolytic and charge-transfer bond-breaking templates.

Args:
    mol_ptr: int address of OpenBabel::OBMol
    given_charge: charge budget to be updated in place and returned
    given_radical: target radical budget used by the stage
)pbdoc",
        py::arg("mol_ptr"),
        py::arg("given_charge") = 0,
        py::arg("given_radical") = 0);

}

}
}
